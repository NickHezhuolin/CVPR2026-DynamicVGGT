import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from dynamicvggt.models.parallel_aggregator import ParallelAggregator
from dynamicvggt.models.aggregator import Aggregator
from dynamicvggt.heads.camera_head import CameraHead
from dynamicvggt.heads.dpt_head import DPTHead

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from typing import Optional, Tuple, List, Any


class DynamicVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        patch_embed="dinov2_vitl14_reg",
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_gs=True,
        enable_future_point=True,
        aa_order=["frame", "global", "TA"],
        load_dino_path=None,
        freeze_vggt=False,
        enable_motion_tokens=True,
        use_camera_enc=False,
    ):
        super().__init__()
        if enable_future_point:
            self.aggregator = ParallelAggregator(
                img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed=patch_embed,
                aa_order=aa_order, load_dino_path=load_dino_path, freeze_vggt=freeze_vggt,
                enable_motion_tokens=enable_motion_tokens, use_camera_enc=use_camera_enc,
            )
        else:
            self.aggregator = Aggregator(
                img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                patch_embed=patch_embed, aa_order=aa_order, load_dino_path=load_dino_path,
            )
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None

        # depth + camera is generally more accurate than the point branch
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None

        # Future point head consumes temporal tokens [B, T, V, P, C]
        self.enable_future_point = enable_future_point
        self.future_point_head = DPTHead(
            dim_in=embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1",
            intermediate_layer_idx=[0, 1, 2, 3],
        ) if enable_future_point else None

        # Initialize future_point_head with point_head weights (for layers with compatible shapes)
        if enable_future_point and enable_point:
            self.init_future_point_head_from_point_head()

        self.enable_gs_head = enable_gs
        if enable_gs:
            from dynamicvggt.heads.gs_dpt_head import GaussianSplatDPTHead
            from dynamicvggt.heads.gaussian_modules.rasterization import GaussianSplatRenderer

            gs_dim = 256
            self.gs_head = GaussianSplatDPTHead(
                dim_in=embed_dim,
                output_dim=2,
                patch_size=patch_size,
                features=gs_dim,
                is_gsdpt=True,
                activation="exp+expp1",
            )
            num_motion_tokens = 16 if enable_motion_tokens else 0
            self.gs_renderer = GaussianSplatRenderer(
                sh_degree=0,
                predict_offset=False,
                predict_residual_sh=True,
                enable_prune=True,
                voxel_size=0.002,
                using_gtcamera_splat=True,
                render_novel_views=True,
                num_motion_tokens=num_motion_tokens,
            )
        else:
            self.gs_head = None

    def init_future_point_head_from_point_head(self):
        """Initialize future_point_head from point_head (callable after checkpoint loading)."""
        self._init_future_point_head_with_point_head()

    def _init_future_point_head_with_point_head(self):
        """
        Initialize future_point_head with point_head weights.
        Copies decoder weights directly; projects/norm use the first embed_dim channels
        from point_head (frame+global concat -> temporal-only input).
        """
        if self.point_head is None or self.future_point_head is None:
            return

        embed_dim = self.future_point_head.norm.normalized_shape[0]

        # LayerNorm(2C) -> LayerNorm(C): use the first half of affine parameters
        self.future_point_head.norm.weight.data.copy_(self.point_head.norm.weight.data[:embed_dim])
        self.future_point_head.norm.bias.data.copy_(self.point_head.norm.bias.data[:embed_dim])

        # Conv2d(in=2C) -> Conv2d(in=C): slice input channels
        for src_proj, dst_proj in zip(self.point_head.projects, self.future_point_head.projects):
            dst_proj.weight.data.copy_(src_proj.weight.data[:, :embed_dim])
            if src_proj.bias is not None and dst_proj.bias is not None:
                dst_proj.bias.data.copy_(src_proj.bias.data)

        for i, (src, dst) in enumerate(zip(self.point_head.resize_layers, self.future_point_head.resize_layers)):
            if isinstance(src, nn.Identity):
                continue  # Skip Identity layers
            dst.load_state_dict(src.state_dict())

        # Copy scratch layers (layer1_rn, layer2_rn, layer3_rn, layer4_rn)
        if hasattr(self.point_head.scratch, 'layer1_rn'):
            self.future_point_head.scratch.layer1_rn.load_state_dict(self.point_head.scratch.layer1_rn.state_dict())
        if hasattr(self.point_head.scratch, 'layer2_rn'):
            self.future_point_head.scratch.layer2_rn.load_state_dict(self.point_head.scratch.layer2_rn.state_dict())
        if hasattr(self.point_head.scratch, 'layer3_rn'):
            self.future_point_head.scratch.layer3_rn.load_state_dict(self.point_head.scratch.layer3_rn.state_dict())
        if hasattr(self.point_head.scratch, 'layer4_rn'):
            self.future_point_head.scratch.layer4_rn.load_state_dict(self.point_head.scratch.layer4_rn.state_dict())

        # Copy refinenet layers
        for i in range(1, 5):
            src_refine = getattr(self.point_head.scratch, f'refinet{i}', None)
            dst_refine = getattr(self.future_point_head.scratch, f'refinet{i}', None)
            if src_refine is not None and dst_refine is not None:
                dst_refine.load_state_dict(src_refine.state_dict())

        # Copy output_conv layers
        if hasattr(self.point_head.scratch, 'output_conv1'):
            self.future_point_head.scratch.output_conv1.load_state_dict(self.point_head.scratch.output_conv1.state_dict())
        if hasattr(self.point_head.scratch, 'output_conv2'):
            self.future_point_head.scratch.output_conv2.load_state_dict(self.point_head.scratch.output_conv2.state_dict())

    def _predict(self, images, batch, predictions):
        """Run the prediction heads on aggregated and temporal tokens."""
        B, T, V, C, H, W = images.shape

        aggregated_tokens_list, patch_start_idx, _, temporal_token_list, ta_patch_start_idx = self.aggregator(
            images, return_cls_token=False
        )

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc_list[-1], images.shape[-2:])
                predictions["extrinsic"] = extrinsic
                predictions["intrinsic"] = intrinsic

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.future_point_head is not None:
                # future_point_head (DPTHead) expects tokens shaped [B, S, P, C]
                _, _, _, P, emb_dim = temporal_token_list[0].shape
                fu_temporal_token_list = [token.reshape(B, T * V, P, emb_dim) for token in temporal_token_list]
                future_pts3d, future_pts3d_conf = self.future_point_head(
                    fu_temporal_token_list, images=images, patch_start_idx=ta_patch_start_idx
                )
                predictions["future_world_points"] = future_pts3d
                predictions["future_world_points_conf"] = future_pts3d_conf

            if self.gs_head is not None:
                # gs_head (GaussianSplatDPTHead) expects tokens shaped [B, T, V, P, C]
                gs_feat, gs_depth, gs_depth_conf, motion_token = self.gs_head(
                    temporal_token_list, images=images, patch_start_idx=ta_patch_start_idx
                )
                predictions["gs_depth"] = gs_depth
                predictions["gs_depth_conf"] = gs_depth_conf
                predictions, render_results = self.gs_renderer(
                    gs_feats=gs_feat,
                    images=images,
                    predictions=predictions,
                    batch=batch,
                    motion_tokens=motion_token,
                )
                predictions["render_results"] = render_results

        return predictions

    def forward(self, batch, query_points: torch.Tensor = None):
        is_multiview_temporal = "input_extrinsics" in batch
        if is_multiview_temporal:
            images = batch["input_images"]
        else:
            images = batch["images"]

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        predictions = {}
        predictions = self._predict(images, batch, predictions)

        # Store the images for visualization during inference
        predictions["images"] = images
        # time_gap controls stage-1 future point supervision (optional)
        predictions["time_gap"] = batch.get("time_gap")

        return predictions

    def inference(self, images: torch.Tensor, batch, query_points: torch.Tensor = None):
        if len(images.shape) == 5:
            images = images.unsqueeze(0)

        predictions = {}
        predictions = self._predict(images, batch, predictions)

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
