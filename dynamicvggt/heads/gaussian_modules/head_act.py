import math
from typing import Callable, Literal, Optional, Tuple, Union, List

import collections.abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from itertools import repeat
from scipy.spatial.transform import Rotation
import numpy as np

def _ntuple(n):
    """
    Creates a parser that converts an input to a tuple of length n.

    Args:
        n (int): Length of the tuple.

    Returns:
        Callable: A function that parses the input into a tuple of length n.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class PluckerEmbedder(nn.Module):
    """
    Convert rays to plucker embedding
    """

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 1,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])

        x, y = torch.meshgrid(
            torch.arange(self.grid_size[1]),
            torch.arange(self.grid_size[0]),
            indexing="xy",
        )
        x = x.float().reshape(1, -1) + 0.5
        y = y.float().reshape(1, -1) + 0.5
        self.register_buffer("x", x)
        self.register_buffer("y", y)

    def forward(
        self,
        intrinsics: Tensor,
        camtoworlds: Tensor,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Tensor:
        assert intrinsics.shape[-2:] == (3, 3), "intrinsics should be (B, 3, 3)"
        assert camtoworlds.shape[-2:] == (4, 4), "camtoworlds should be (B, 4, 4)"
        intrinsics_shape = intrinsics.shape
        intrinsics = intrinsics.reshape(-1, 3, 3)
        camtoworlds = camtoworlds.reshape(-1, 4, 4)
        if image_size is not None:
            image_size = to_2tuple(image_size)
        else:
            image_size = self.img_size
        if patch_size is not None:
            patch_size = to_2tuple(patch_size)
        else:
            patch_size = self.patch_size

        grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])

        if grid_size != self.grid_size:
            grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])
            x, y = torch.meshgrid(
                torch.arange(grid_size[1]),
                torch.arange(grid_size[0]),
                indexing="xy",
            )
            x = x.float().reshape(1, -1) + 0.5
            y = y.float().reshape(1, -1) + 0.5
            x = x.to(intrinsics.device)
            y = y.to(intrinsics.device)
            intrinsics = intrinsics.clone()
            # intrinsics should be scaled to the grid size
            intrinsics[..., 0, 0] = intrinsics[..., 0, 0] / patch_size[1]
            intrinsics[..., 0, 2] = intrinsics[..., 0, 2] / patch_size[1]
            intrinsics[..., 1, 1] = intrinsics[..., 1, 1] / patch_size[0]
            intrinsics[..., 1, 2] = intrinsics[..., 1, 2] / patch_size[0]
        else:
            x, y = self.x, self.y

        x = x.repeat(intrinsics.size(0), 1)
        y = y.repeat(intrinsics.size(0), 1)
        camera_dirs = torch.nn.functional.pad(
            torch.stack(
                [
                    (x - intrinsics[:, 0, 2][..., None] + 0.5) / intrinsics[:, 0, 0][..., None],
                    (y - intrinsics[:, 1, 2][..., None] + 0.5) / intrinsics[:, 1, 1][..., None],
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )
        directions = torch.sum(camera_dirs[:, :, None, :] * camtoworlds[:, None, :3, :3], dim=-1)
        origins = torch.broadcast_to(camtoworlds[:, :3, -1].unsqueeze(1), directions.shape)
        direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
        viewdirs = directions / (direction_norm + 1e-8)
        cross_prod = torch.cross(origins, viewdirs, dim=-1)
        plucker = torch.cat((cross_prod, viewdirs), dim=-1)
        origins = rearrange(origins, "b (h w) c -> b h w c", h=grid_size[0])
        viewdirs = rearrange(viewdirs, "b (h w) c -> b h w c", h=grid_size[0])
        directions = rearrange(directions, "b (h w) c -> b h w c", h=grid_size[0])
        plucker = rearrange(plucker, "b (h w) c -> b h w c", h=grid_size[0])
        return {
            "origins": origins.view(*intrinsics_shape[:-2], *grid_size, 3),
            "viewdirs": viewdirs.view(*intrinsics_shape[:-2], *grid_size, 3),
            "dirs": directions.view(*intrinsics_shape[:-2], *grid_size, 3),
            "plucker": plucker.view(*intrinsics_shape[:-2], *grid_size, 6),
        }

def modulate(x, shift=None, scale=None):
    if shift is None and scale is None:
        return x
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class ModulatedLinearLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, condition_channels=768, out_channels=3):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_channels, 2 * hidden_channels, bias=True)
        )
        self.condition_mapping = nn.Linear(condition_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, c):
        x = self.linear(x)
        c = self.condition_mapping(c.squeeze(1))
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x_shape = x.shape
        x = modulate(self.norm(x.reshape(x_shape[0], -1, x.shape[-1])), shift, scale)
        x = self.output(x)
        x = x.reshape(*x_shape[:-1], -1)
        return x

def check_results(result_dict) -> bool:
    assert "rgb_key" in result_dict, "rgb_key not found in result_dict"
    assert "depth_key" in result_dict, "depth_key not found in result_dict"
    assert "alpha_key" in result_dict, "alpha_key not found in result_dict"
    assert "flow_key" in result_dict, "flow_key not found in result_dict"
    assert "decoder_depth_key" in result_dict, "decoder_depth_key not found in result_dict"
    assert "decoder_alpha_key" in result_dict, "decoder_alpha_key not found in result_dict"
    assert "decoder_flow_key" in result_dict, "decoder_flow_key not found in result_dict"
    return True

class DummyDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(DummyDecoder, self).__init__()

    def forward(self, render_results):
        if not check_results(render_results):
            raise ValueError("Invalid result dict")
        return render_results

class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim=None, out_dim=None, act_layer=nn.GELU):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def compute_depth_from_points(self, points, origins, directions):
    """从3D点云计算深度（沿射线方向的距离）"""
    # points: [B, T, V, H, W, 3]
    # origins: [B, T, V, H, W, 3] 
    # directions: [B, T, V, H, W, 3] (单位向量)
    
    # 计算从射线原点到3D点的向量
    ray_to_point = points - origins  # [B, T, V, H, W, 3]
    
    # 沿射线方向的投影距离
    depths = torch.sum(ray_to_point * directions, dim=-1, keepdim=True)  # [B, T, V, H, W, 1]
    
    return depths.squeeze(-1)  # [B, T, V, H, W]

def prepare_data_dict(data_dict, predictions, batch, time_data):
    """
    准备渲染数据，基于context帧插值得到所有帧的外参
    
    Args:
        data_dict: 基础数据字典
        predictions: 模型预测结果
        batch: 输入batch数据
        time_data: 时间数据
        
    Returns:
        包含插值后外参的完整data_dict
    """
    B, T, V, H, W = data_dict["batch_shape"]
    device = predictions["extrinsic"].device
    
    # 1. 基于context帧的外参插值得到所有帧
    context_extrinsics = predictions["extrinsic"].view(B, T, V, 3, 4)
    all_extrinsics = interpolate_extrinsics(context_extrinsics)  # [B, 2*T, V, 3, 4]
    
    # 2. 分离context和target帧
    render_context_extrinsics = all_extrinsics[:, 0::2]  # 024帧
    render_target_extrinsics = all_extrinsics[:, 1::2]   # 135帧
    
    # 3. 准备完整的数据字典
    full_data_dict = {
        # 外参（使用插值结果）
        "context_camtoworlds": render_context_extrinsics,  # [B, T, V, 3, 4]
        "target_camtoworlds": render_target_extrinsics,    # [B, T, V, 3, 4]
        "render_camtoworlds": all_extrinsics,              # [B, 2*T, V, 3, 4]
        
        # 内参（保持不变）
        "context_intrinsics": predictions["intrinsic"].view(B, T, V, 3, 3),
        "target_intrinsics": batch["target_intrinsics"],
        "render_intrinsics": interpolate_intrinsics(
            predictions["intrinsic"].view(B, T, V, 3, 3), 
            batch["target_intrinsics"]
        ),
        
        # 图像尺寸
        "height": H,
        "width": W,
        
        # 时间信息
        "context_time": time_data["context_time"],
        "target_time": time_data["target_time"], 
        "render_time": interpolate_time(time_data),
        "timespan": time_data["timespan"],

        "is_interpolated": True  # 标记使用了插值
    }
    
    return full_data_dict

def interpolate_extrinsics(extrinsics, method='slerp'):
    """
    基于已知帧的外参插值计算中间帧的外参
    
    Args:
        extrinsics: [B, T, V, 3, 4] 已知帧的外参（如024帧）
        method: 插值方法 ('slerp', 'linear', 'bezier')
        
    Returns:
        interpolated: [B, T, V, 3, 4] 插值得到的所有帧外参（012345帧）
    """
    B, T_ctx, V, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype
    
    # 总帧数（已知帧 + 插值帧）
    total_frames = 2 * T_ctx  # 比如从3帧插值到6帧
    
    interpolated = torch.zeros(B, total_frames, V, 3, 4, device=device, dtype=dtype)
    
    # 已知帧放在偶数位置
    interpolated[:, 0::2] = extrinsics
    
    # 对每个batch和view分别插值
    for b in range(B):
        for v in range(V):
            # 提取该view的所有已知外参
            known_extrinsics = extrinsics[b, :, v]  # [T_ctx, 3, 4]
            
            # 插值得到所有帧
            all_extrinsics = _interpolate_single_sequence(known_extrinsics, method)
            
            interpolated[b, :, v] = all_extrinsics
    
    return interpolated

def _interpolate_single_sequence(extrinsics, method='slerp'):
    """
    对单个相机序列进行插值
    """
    T_ctx = extrinsics.shape[0]
    total_frames = 2 * T_ctx
    
    # 分离旋转和平移
    rotations = extrinsics[:, :3, :3]  # [T_ctx, 3, 3]
    translations = extrinsics[:, :3, 3]  # [T_ctx, 3]
    
    # 插值旋转（使用SLERP）
    interp_rotations = _slerp_rotations(rotations, total_frames)
    
    # 插值平移（使用线性插值）
    interp_translations = _linear_interpolate(translations, total_frames)
    
    # 重新组合外参矩阵
    interp_extrinsics = torch.eye(4, device=extrinsics.device).unsqueeze(0).repeat(total_frames, 1, 1)
    interp_extrinsics[:, :3, :3] = interp_rotations
    interp_extrinsics[:, :3, 3] = interp_translations
    
    return interp_extrinsics[:, :3, :4]  # 返回3x4格式

def _slerp_rotations(rotations, total_frames):
    """
    球面线性插值旋转矩阵
    """
    T_ctx = rotations.shape[0]
    interp_rotations = []
    
    for i in range(T_ctx - 1):
        # 当前帧和下一帧
        R1 = rotations[i].detach().cpu().numpy()
        R2 = rotations[i+1].detach().cpu().numpy()
        
        # 转换为四元数
        rot1 = Rotation.from_matrix(R1)
        rot2 = Rotation.from_matrix(R2)
        quat1 = rot1.as_quat()
        quat2 = rot2.as_quat()

        # 手动实现 SLERP
        def slerp_quaternion(q1, q2, t):
            # 归一化四元数
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)
            
            # 计算点积
            dot = np.dot(q1, q2)
            
            # 如果点积为负，取反其中一个四元数以保证走最短路径
            if dot < 0.0:
                q2 = -q2
                dot = -dot
            
            # 如果四元数非常接近，使用线性插值避免除零
            if dot > 0.9995:
                result = q1 + t * (q2 - q1)
                return result / np.linalg.norm(result)
            
            # 计算角度
            theta_0 = np.arccos(dot)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            
            # SLERP 公式
            s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s2 = sin_theta / sin_theta_0
            result = s1 * q1 + s2 * q2
            return result / np.linalg.norm(result)
        
        # 在两帧之间插值
        for j in range(2):  # 插值2个中间帧（0->1, 1->2）
            alpha = (j + 1) / 2.0  # 0.5, 1.0
            interp_quat = slerp_quaternion(quat1, quat2, alpha)
            interp_rot = Rotation.from_quat(interp_quat).as_matrix()
            
            interp_tensor = torch.tensor(interp_rot, device=rotations.device, dtype=rotations.dtype)
            interp_rotations.append(interp_tensor.unsqueeze(0))  # 添加batch维度 [1, 3, 3]
    
    # 添加首尾帧
    interp_rotations = [rotations[0].unsqueeze(0)] + interp_rotations + [rotations[-1].unsqueeze(0)]
    return torch.cat(interp_rotations, dim=0)

def _linear_interpolate(translations, total_frames):
    """
    线性插值平移向量
    """
    T_ctx = translations.shape[0]
    interp_translations = []
    
    for i in range(T_ctx - 1):
        start_pos = translations[i]
        end_pos = translations[i+1]
        
        for j in range(2):  # 插值2个中间位置
            alpha = (j + 1) / 2.0
            interp_pos = start_pos * (1 - alpha) + end_pos * alpha
            interp_translations.append(interp_pos.unsqueeze(0))
    
    # 添加首尾帧
    interp_translations = [translations[0].unsqueeze(0)] + interp_translations + [translations[-1].unsqueeze(0)]
    return torch.cat(interp_translations, dim=0)

def interpolate_intrinsics(context_intrinsics, target_intrinsics):
    """
    插值内参矩阵（通常变化不大，可以简单处理）
    """
    B, T, V, _, _ = context_intrinsics.shape
    device = context_intrinsics.device
    
    # 创建所有帧的内参
    all_intrinsics = torch.zeros(B, 2*T, V, 3, 3, device=device)
    
    # context帧使用预测值，target帧使用真实值（或插值）
    all_intrinsics[:, 0::2] = context_intrinsics
    all_intrinsics[:, 1::2] = target_intrinsics
    
    return all_intrinsics

def interpolate_time(time_data):
    """
    插值时间序列
    """
    context_time = time_data["context_time"]  # [B, T, V]
    target_time = time_data["target_time"]    # [B, T, V]
    B, T, V= context_time.shape
    
    # 创建交错的时间序列
    render_time = torch.zeros(B, 2*T, V, device=context_time.device)
    render_time[:, 0::2, :] = context_time
    render_time[:, 1::2, :] = target_time
    
    return render_time

def prepare_time_data(batch):
    """
    Convert input_ids/target_ids/time_gap to normalized context_time, target_time, timespan.
    
    Args:
        batch: dict with keys:
            - "input_ids": [B, T_ctx]
            - "target_ids": [B, T_tgt]
            - "time_gap": [B]
    
    Returns:
        dict with:
            - "context_time": [B, T_ctx, V]
            - "target_time": [B, T_tgt, V]
            - "timespan": [B]
    """
    input_ids = batch["input_ids"]      # [B, T_ctx]
    target_ids = batch["target_ids"]    # [B, T_tgt]
    time_gap = batch["time_gap"]        # [B]
    num_views = batch["view_num"]      # [B]
    actual_num_views = num_views[0].item()

    # Combine all frame IDs to compute global min/max
    all_ids = torch.cat([input_ids, target_ids], dim=1)  # [B, T_ctx + T_tgt]
    min_id = all_ids.min(dim=1, keepdim=True).values     # [B, 1]
    max_id = all_ids.max(dim=1, keepdim=True).values     # [B, 1]

    # Physical time = frame_id * time_gap
    ctx_physical = input_ids * time_gap.unsqueeze(1)     # [B, T_ctx]
    tgt_physical = target_ids * time_gap.unsqueeze(1)    # [B, T_tgt]
    min_time = min_id * time_gap.unsqueeze(1)            # [B, 1]
    max_time = max_id * time_gap.unsqueeze(1)            # [B, 1]

    # Timespan = max_time - min_time
    timespan = (max_time - min_time).squeeze(1)          # [B]

    # Normalize to [0, 1]
    context_time = (ctx_physical - min_time) / timespan.unsqueeze(1)  # [B, T_ctx]
    target_time = (tgt_physical - min_time) / timespan.unsqueeze(1)   # [B, T_tgt]

    # Handle edge case: timespan=0 (all frames same)
    timespan = torch.where(timespan == 0, torch.ones_like(timespan), timespan)
    context_time = torch.where(timespan.unsqueeze(1) == 0, torch.zeros_like(context_time), context_time)
    target_time = torch.where(timespan.unsqueeze(1) == 0, torch.zeros_like(target_time), target_time)

    # 扩展到多视角
    context_time = context_time.unsqueeze(-1).repeat(1, 1, actual_num_views)  # [B, T_ctx, V]
    target_time = target_time.unsqueeze(-1).repeat(1, 1, actual_num_views)    # [B, T_tgt, V]

    return {
        "context_time": context_time,  # [B, T_ctx, V]
        "target_time": target_time,    # [B, T_tgt, V]
        "timespan": timespan           # [B]
    }

def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )

def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch

def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

# Reuse the same supporting classes from DPTHead
class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output
