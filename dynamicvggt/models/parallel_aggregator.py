import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import *

from vggt.layers import PatchEmbed
from vggt.layers.block import Block #attn
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

# v2: camera_encoder & motion_encoder
from dynamicvggt.aux_encoder.camera_encoder import CameraEnc
# from dynamicvggt.aux_encoder.motion_encoder import MotionEnc

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class ParallelAggregator(nn.Module):
    """
    Parallel Aggregator with frame/global attention and temporal attention branches.
    """
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global", "TA"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        load_dino_path=None,
        freeze_vggt=True,  # 关键：是否冻结 VG-GT 权重
        enable_motion_tokens=True,
        use_camera_enc=True,
        camera_enc_dim_in=9,
        ta_block_idx=[4, 11, 17, 23],
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed, img_size, patch_size, num_register_tokens, 
            embed_dim=embed_dim,
            load_dino_path=load_dino_path
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # VG-GT Backbone (AA + GA blocks) - will be frozen
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.ta_bloack_idx = ta_block_idx
        # 根据 aa_order 中的模块类型创建相应的 blocks
        if any(mod in aa_order for mod in ["TA"]):
            # Time Attention
            self.time_blocks = nn.ModuleList([
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                ) for _ in range(len(ta_block_idx))
            ])
        else:
            # 没有时序模块
            self.time_blocks = None

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.freeze_vggt = freeze_vggt

        # Initialize time_blocks with frame_blocks weights
        if self.time_blocks is not None and len(ta_block_idx) > 0:
            self._init_time_blocks_with_frame_blocks()

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # new add
        self.use_camera_enc = use_camera_enc
        if self.use_camera_enc:
            self.camera_enc = CameraEnc(dim_out=embed_dim, dim_in=camera_enc_dim_in)

        # self.use_camera_enc = use_motion_enc
        # if self.use_motion_enc: 
        #     self.motion_enc = MotionEnc(dim_out=embed_dim)

        # Special tokens (now designed for (T, V) structure)
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))  # [1, 2, 1, C]
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))  # [1, 2, R, C]
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        if enable_motion_tokens:
            self.enable_motion_tokens = enable_motion_tokens
            self.num_motion_tokens = 16
            self.motion_token = nn.Parameter(torch.randn(1, 2, self.num_motion_tokens, embed_dim))  # [1, num, C]
            self.ta_patch_start_idx = self.num_motion_tokens
            nn.init.normal_(self.motion_token, std=1e-6)
        else:
            self.enable_motion_tokens = None
            self.ta_patch_start_idx = 0

        # Register normalization constants
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).reshape(1, 1, 3, 1, 1), persistent=False)
        
        self.use_reentrant = False

        # Freeze VG-GT backbone if requested
        if freeze_vggt:
            self._freeze_vggt_backbone()

    def _freeze_vggt_backbone(self):
        """Freeze all parameters in frame_blocks and global_blocks"""
        for param in self.frame_blocks.parameters():
            param.requires_grad = False
        for param in self.global_blocks.parameters():
            param.requires_grad = False
        logger.info("Frozen VG-GT backbone (frame_blocks + global_blocks)")

    def _init_time_blocks_with_frame_blocks(self):
        """
        Initialize time_blocks with frame_blocks weights at corresponding ta_block_idx positions.
        This helps with training stability by initializing temporal attention blocks with
        pretrained spatial attention weights.
        """
        for i, block_idx in enumerate(self.ta_bloack_idx):
            if block_idx < len(self.frame_blocks):
                src_block = self.frame_blocks[block_idx]
                dst_block = self.time_blocks[i]
                dst_block.load_state_dict(src_block.state_dict())
                logger.info(f"Initialized time_blocks[{i}] with frame_blocks[{block_idx}] weights")

    def _get_camera_token(self, B, S, ext=None, ixt=None, image_size=None):
        """
        根据是否有内外参动态生成camera_token
        Args:
            B: batch size
            S: sequence length (T*V)
            ext: 相机外参 (B, T, V, 4, 4) 或 None
            ixt: 相机内参 (B, T, V, 3, 3) 或 None
            image_size: 图像尺寸 (B, T, V, 2) 或 None
        Returns:
            camera_token: (B*S, 1, C)
        """
        if self.use_camera_enc and ext is not None and ixt is not None and image_size is not None:
            B_total, T, V, *ext_shape = ext.shape
            ext_flat = ext.view(B_total * T * V, *ext_shape)
            ixt_flat = ixt.view(B_total * T * V, *ixt.shape[-2:])
            image_size_flat = image_size.view(B_total * T * V, *image_size.shape[-1:])
            camera_features = self.camera_enc(ext_flat, ixt_flat, image_size_flat)
            camera_features = camera_features.unsqueeze(1)  # (B*S, 1, C)
            return camera_features
        else:
            return slice_expand_and_flatten(self.camera_token, B, S)

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        load_dino_path=None
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        elif "dinov2" in patch_embed:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            if load_dino_path is not None:
                pretrained = torch.load(load_dino_path)
                msg = self.patch_embed.load_state_dict(pretrained, strict=False)
                print(f"Loaded DINOv2 ViT-L/14 weights with msg: {msg}")

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)
        else:
            raise ValueError(
                f"Unsupported patch_embed: {patch_embed}. Use dinov2_* or conv."
            )

    def forward(
        self,
        images: torch.Tensor,
        ext=None,  # 新增：相机外参 (B, T, V, 4, 4)
        ixt=None,  # 新增：相机内参 (B, T, V, 3, 3)
        image_size=None,  # 新增：图像尺寸 (B, T, V, 2)
        return_cls_token=False
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, T, V, 3, H, W], in range [0, 1].
                B: batch size, Time * View = S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, T, V, C_in, H, W = images.shape
        S = T*V

        # Normalize images
        images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)
        
        # Reshape for patch embedding: (B*T*V, C, H, W)
        images_flat = images.view(B * T * V, C_in, H, W)
        patch_tokens = self.patch_embed(images_flat)

        if isinstance(patch_tokens, dict):
            if return_cls_token:
                cls_tokens = patch_tokens['x_norm_clstoken']  # (B*T*V, C)
                cls_token_list = cls_tokens.view(B, S, -1)  # (B, T*V, C)
            patch_tokens = patch_tokens["x_norm_patchtokens"]  # (B*T*V, P, C)

        _, P_ori, C = patch_tokens.shape

        camera_token = self._get_camera_token(B, S, ext, ixt, image_size)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        if self.enable_motion_tokens:
            motion_token = slice_expand_and_flatten(self.motion_token, B, S)
            motion_token = motion_token.view(B, T, V, self.num_motion_tokens, C)
        else:
            motion_token = None

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        # Get positional embeddings
        pos = None
        if self.rope is not None:
            # Get positions for (B*T*V, H//patch, W//patch, 2)
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        time_idx = 0
        output_list = []
        temporal_token_list = []

        sp_tokens = tokens   # [B, T, V, 1 + R + P, C]
        future_tokens = None # Will store [B, T, V, M+P, C]

        for bk_n in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    sp_tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        sp_tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    sp_tokens, global_idx, global_intermediates = self._process_global_attention(
                        sp_tokens, B, S, P, C, global_idx, pos=pos
                    )
                elif attn_type == "TA":
                    if bk_n in self.ta_bloack_idx:
                        _, time_idx, ta_intermediates, future_tokens = self._process_time_attention(
                            sp_tokens, B, T, V, P, C, time_idx, future_tokens=future_tokens, motion_token=motion_token, patch_len=P_ori
                        )
                    else:
                        continue
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            
            for i in range(len(frame_intermediates)):
                # for new future_point_head
                # concat frame, global and temporal intermediates, [B, T, V, P, C]
                if bk_n in self.ta_bloack_idx:
                    temporal_token_list.append(ta_intermediates[i])

                # for vggt head
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
            
            del concat_inter
            del frame_intermediates
            del global_intermediates
            if bk_n in self.ta_bloack_idx:
                del ta_intermediates

        if return_cls_token:
            return output_list, self.patch_start_idx, cls_token_list, temporal_token_list, self.ta_patch_start_idx
        else:
            return output_list, self.patch_start_idx, None, temporal_token_list, self.ta_patch_start_idx

    def _process_time_attention(self, tokens, B, T, V, P, C, time_idx, future_tokens=None, motion_token=None, patch_len=480):
        """
        Process time attention blocks with parallel state passing.
        
        Args:
            sp_tokens: (B, T, V, P, C) - current spatial tokens
            B, T, V, P, C: batch, time, view, patch, channel dims   P = 1+R+P
            time_idx: current time attention block index
            future_tokens: previous time features from earlier blocks (optional)
        
        Returns:
            tuple: (updated_sp_tokens, new_time_idx, intermediates, updated_future_tokens)
        """
        # Handle input shape
        tokens = tokens.view(B, T, V, P, C)
        sp_patches = tokens[:, :, :, self.patch_start_idx:, :]

        # === Step 2: 准备 TA 输入 ===
        if self.enable_motion_tokens:
            if future_tokens is None:
                # First TA layer: use initial motion_token
                current_motion = motion_token  # [B, T, V, M, C]
                current_patches = sp_patches  # [B, T, V, P, C]
            else:
                # Subsequent TA layer: 
                # - motion part: carry over from prev_future_tokens
                # - patches part: sp_patches_4d + prev_future_tokens' patches
                prev_motion = future_tokens[:, :, :, :self.num_motion_tokens, :]  # [B, T, V, M, C]
                prev_patches = future_tokens[:, :, :, self.num_motion_tokens:, :]  # [B, T, V, P, C]
                current_motion = prev_motion
                current_patches = sp_patches + prev_patches
            
            # Concatenate: [motion; patches] -> [B, T, V, M+P, C]
            ta_input = torch.cat([current_motion, current_patches], dim=3)
            L = self.num_motion_tokens + patch_len
        else:
            # Without motion tokens
            if future_tokens is None:
                ta_input = sp_patches
            else:
                prev_patches = future_tokens
                ta_input = sp_patches + prev_patches
            L = patch_len

        # Extract temporal dimension: (B*V*P, T, C)
        temporal_input = ta_input.permute(0, 2, 3, 1, 4).reshape(B * V * L, T, C)
        time_pos = self._create_time_positional_embeddings(B, V, L, T, ta_input.device)
        
        # Apply time attention block
        if self.training:
            temporal_output = checkpoint(
                self.time_blocks[time_idx],
                temporal_input,
                time_pos,
                use_reentrant=self.use_reentrant
            )
        else:
            temporal_output = self.time_blocks[time_idx](temporal_input, pos=time_pos)
        
        # Update future_tokens for next iteration
        future_tokens = temporal_output.reshape(B, V, L, T, C).permute(0, 3, 1, 2, 4)
        
        # Create intermediates list
        intermediates = [future_tokens]  # List of (B, T, V, P, C)
        
        return tokens, time_idx + 1, intermediates, future_tokens

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:

        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _create_time_positional_embeddings(self, B, V, P, T, device):
        """
        Create positional embeddings for time dimension for RoPE.
        
        Args:
            B, V, P, T: batch, view, patch, time dimensions
            device: target device
        
        Returns:
            torch.Tensor: (B*V*P, T, 2) time positional embeddings
        """
        # For RoPE, we need (T, 2) where 2 is for cos/sin components
        # But we need to broadcast to (B*V*P, T, 2)
        
        if self.position_getter is not None:
            # Use the same position getter as spatial positions, but for time dimension
            # Create positions for sequence length T
            time_pos = self.position_getter(1, T, 1, device=device)  # (1, T, 2)
            time_pos = time_pos.expand(B * V * P, -1, -1)  # (B*V*P, T, 2)
        else:
            # Fallback: simple sinusoidal embeddings
            position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)  # (T, 1)
            div_term = torch.exp(
                torch.arange(0, 2, dtype=torch.float, device=device) * 
                (-torch.log(torch.tensor(10000.0, device=device)) / 2)
            )  # (2,)
            
            pe = torch.zeros(T, 2, device=device)  # (T, 2)
            pe[:, 0] = torch.sin(position * div_term[0])  # cos component
            pe[:, 1] = torch.cos(position * div_term[1])  # sin component
            
            time_pos = pe.unsqueeze(0).expand(B * V * P, -1, -1)  # (B*V*P, T, 2)
        
        return time_pos

def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined
