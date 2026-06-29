import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Union
from dynamicvggt.layers.layers import Mlp
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vggt.utils.geometry import affine_inverse

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        # Debug breakpoint removed for production
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = self.rope(q, pos) if self.rope is not None else q
        k = self.rope(k, pos) if self.rope is not None else k
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.sample_drop_ratio = 0.0  # Equivalent to always having drop_path=0

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, attn_mask=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        # drop_path is always 0, so always take the else branch
        x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
        x = x + ffn_residual_func(x)
        return x

class CameraEnc(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.
    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    输入:
        ext (B,4,4) 相机外参矩阵(世界→相机) 
        ixt (B,3,3) 相机内参矩阵 
        image_size (B,2) [height, width] 
    中间特征: 
        pose_encoding (B,9) 几何编码向量(由extri_intri_to_pose_encoding生成)
    输出:
        pose_tokens (B,1024) 相机语义特征向量(可用于3D任务)
    """

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        target_dim: int = 9,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            drop=0,
        )

    def forward(
        self,
        ext,
        ixt,
        image_size,
    ) -> tuple:
        c2ws = affine_inverse(ext)
        pose_encoding = extri_intri_to_pose_encoding(
            c2ws,
            ixt,
            image_size,
        )
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens
