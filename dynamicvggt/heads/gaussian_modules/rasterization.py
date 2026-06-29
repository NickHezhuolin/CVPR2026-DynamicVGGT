from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from einops import rearrange

try:
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
except ImportError:
    rasterization = None
    DefaultStrategy = None

from .head_act import interpolate_intrinsics, prepare_time_data, LayerNorm2d, Mlp, interpolate_time

from pathlib import Path
from plyfile import PlyData, PlyElement

# -- vis --
def save_gs_ply(path: str,
                means: torch.Tensor,
                scales: torch.Tensor,
                rotations: torch.Tensor,
                rgbs: torch.Tensor,
                opacities: torch.Tensor) -> None:
    """
    Export Gaussian splat data to PLY format.
    
    Args:
        path: Output PLY file path
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        rotations: Gaussian rotations as quaternions [N, 4]
        rgbs: RGB colors [N, 3]
        opacities: Opacity values [N]
    """
    # 确保所有张量在CPU上
    means = means.detach().cpu().float()
    scales = scales.detach().cpu().float()
    rotations = rotations.detach().cpu().float()
    rgbs = rgbs.detach().cpu().float()
    opacities = opacities.detach().cpu().float()

    # 数据清洗和验证
    def clean_tensor(tensor, name):
        # 检查NaN和Inf
        if torch.isnan(tensor).any():
            print(f"警告: {name} 包含NaN值，进行清理")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        
        if torch.isinf(tensor).any():
            print(f"警告: {name} 包含Inf值，进行清理")
            tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
        
        return tensor

    # 清理所有张量
    means = clean_tensor(means, "means")
    scales = clean_tensor(scales, "scales")
    rotations = clean_tensor(rotations, "rotations")
    rgbs = clean_tensor(rgbs, "rgbs")
    opacities = clean_tensor(opacities, "opacities")

    # 确保颜色在有效范围内 [0, 1]
    rgbs = torch.clamp(rgbs, 0.0, 1.0)

    # 过滤异常尺度的高斯
    scale_norms = torch.norm(scales, dim=1)
    scale_threshold = torch.quantile(scale_norms, 0.95)
    filter_mask = scale_norms <= scale_threshold

    # 如果过滤后点数太少，使用宽松的阈值
    if filter_mask.sum() < len(means) * 0.5:
        print(f"警告: 过滤过多点，使用宽松阈值")
        scale_threshold = torch.quantile(scale_norms, 0.98)
        filter_mask = scale_norms <= scale_threshold

    print(f"原始点数: {len(means)}, 过滤后点数: {filter_mask.sum()}")

    # 应用过滤
    means = means[filter_mask]
    scales = scales[filter_mask]
    rotations = rotations[filter_mask]
    rgbs = rgbs[filter_mask]
    opacities = opacities[filter_mask]

    # 检查过滤后是否还有数据
    if len(means) == 0:
        print("错误: 过滤后没有剩余的点")
        return

    # 数据统计
    print(f"数据统计:")
    print(f"  位置范围: X[{means[:,0].min():.3f}, {means[:,0].max():.3f}], "
          f"Y[{means[:,1].min():.3f}, {means[:,1].max():.3f}], "
          f"Z[{means[:,2].min():.3f}, {means[:,2].max():.3f}]")
    print(f"  颜色范围: R[{rgbs[:,0].min():.3f}, {rgbs[:,0].max():.3f}], "
          f"G[{rgbs[:,1].min():.3f}, {rgbs[:,1].max():.3f}], "
          f"B[{rgbs[:,2].min():.3f}, {rgbs[:,2].max():.3f}]")
    print(f"  尺度范围: [{scales.min():.3f}, {scales.max():.3f}]")
    print(f"  不透明度范围: [{opacities.min():.3f}, {opacities.max():.3f}]")

    # 构建属性名称
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")

    # 准备数据 - 转换为numpy并确保连续性
    try:
        means_np = means.numpy().astype(np.float32)
        zeros_np = np.zeros_like(means_np, dtype=np.float32)
        rgbs_np = rgbs.numpy().astype(np.float32)
        opacities_np = opacities.numpy().astype(np.float32).reshape(-1, 1)
        scales_np = scales.log().numpy().astype(np.float32)
        rotations_np = rotations.numpy().astype(np.float32)

        # 检查所有数组的形状
        print(f"数组形状检查:")
        print(f"  means: {means_np.shape}")
        print(f"  zeros: {zeros_np.shape}")
        print(f"  rgbs: {rgbs_np.shape}")
        print(f"  opacities: {opacities_np.shape}")
        print(f"  scales: {scales_np.shape}")
        print(f"  rotations: {rotations_np.shape}")

        # 连接所有属性
        attributes_data = np.concatenate([
            means_np, zeros_np, rgbs_np, opacities_np, scales_np, rotations_np
        ], axis=1)

        # 创建PLY数据结构
        dtype_full = [(attribute, "f4") for attribute in attributes]
        elements = np.empty(attributes_data.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes_data))

        # 写入PLY文件
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
        print(f"✅ 成功导出 {len(means)} 个高斯到 {path}")

    except Exception as e:
        print(f"❌ 导出PLY失败: {e}")
        # 尝试简化版本
        save_simple_ply(path, means, rgbs)

def save_simple_ply(path: str, means: torch.Tensor, rgbs: torch.Tensor):
    """简化版本的PLY导出，只包含位置和颜色"""
    means_np = means.detach().cpu().numpy().astype(np.float32)
    rgbs_np = rgbs.detach().cpu().numpy()
    
    # 确保颜色在0-255范围内
    rgbs_uint8 = (np.clip(rgbs_np, 0, 1) * 255).astype(np.uint8)
    
    # 创建简单的顶点数据
    vertices = np.zeros(means_np.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertices['x'] = means_np[:, 0]
    vertices['y'] = means_np[:, 1]
    vertices['z'] = means_np[:, 2]
    vertices['red'] = rgbs_uint8[:, 0]
    vertices['green'] = rgbs_uint8[:, 1]
    vertices['blue'] = rgbs_uint8[:, 2]
    
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(path)
    print(f"✅ 简化版PLY导出成功: {path}")

# --- Projections ---
def homogenize_points(points):
    """Append a '1' along the final dimension of the tensor (i.e. convert xyz->xyz1)"""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]

def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum('b v i j , h w j -> b v h w i', intrinsics.inverse(), pixel_space_points)
    camera_space_points = camera_space_points * depth
    return camera_space_points

def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum('b v i j , b v h w j -> b v h w i', c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: World space points with shape (b, v1, v2, h, w, 2).
    """
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum('b u i j , b v u h w j -> b v u h w i', intrinsics, camera_space_points)
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to pixel space points.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum('b u i j , b v h w j -> b v u h w i', c2w.inverse(), world_space_points)
    return camera_space_points[..., :3]

def unproject_depth(depth, intrinsics, c2w):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """

    # Compute indices of pixels
    h, w = depth.shape[-3], depth.shape[-2]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )  # (h, w), (h, w)

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points

@torch.no_grad()
def calculate_in_frustum_mask(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2):
    """
    A function that takes in the depth, intrinsics and c2w matrices of two sets
    of views, and then works out which of the pixels in the first set of views
    has a direct corresponding pixel in any of views in the second set

    Args:
        depth_1: (b, v1, h, w)
        intrinsics_1: (b, v1, 3, 3)
        c2w_1: (b, v1, 4, 4)
        depth_2: (b, v2, h, w)
        intrinsics_2: (b, v2, 3, 3)
        c2w_2: (b, v2, 4, 4)

    Returns:
        torch.Tensor: valid mask with shape (b, v1, v2, h, w).
    """

    _, v1, h, w = depth_1.shape
    _, v2, _, _ = depth_2.shape

    # Unproject the depth to get the 3D points in world space
    points_3d = unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)  # (b, v1, h, w, 3)

    # Project the 3D points into the pixel space of all the second views simultaneously
    camera_points = world_space_to_camera_space(points_3d, c2w_2)  # (b, v1, v2, h, w, 3)
    points_2d = camera_space_to_pixel_space(camera_points, intrinsics_2)  # (b, v1, v2, h, w, 2)

    # Calculate the depth of each point
    rendered_depth = camera_points[..., 2]  # (b, v1, v2, h, w)

    # We use three conditions to determine if a point should be masked

    # Condition 1: Check if the points are in the frustum of any of the v2 views
    in_frustum_mask = (
        (points_2d[..., 0] > 0) &
        (points_2d[..., 0] < w) &
        (points_2d[..., 1] > 0) &
        (points_2d[..., 1] < h)
    )  # (b, v1, v2, h, w)
    in_frustum_mask = in_frustum_mask.any(dim=-3)  # (b, v1, h, w)

    # Condition 2: Check if the points have non-zero (i.e. valid) depth in the input view
    non_zero_depth = depth_1 > 1e-6

    # Condition 3: Check if the points have matching depth to any of the v2
    # views torch.nn.functional.grid_sample expects the input coordinates to
    # be normalized to the range [-1, 1], so we normalize first
    points_2d[..., 0] /= w
    points_2d[..., 1] /= h
    points_2d = points_2d * 2 - 1
    matching_depth = torch.ones_like(rendered_depth, dtype=torch.bool)
    for b in range(depth_1.shape[0]):
        for i in range(v1):
            for j in range(v2):
                depth = einops.rearrange(depth_2[b, j], 'h w -> 1 1 h w')
                coords = einops.rearrange(points_2d[b, i, j], 'h w c -> 1 h w c')
                sampled_depths = torch.nn.functional.grid_sample(depth, coords, align_corners=False)[0, 0]
                matching_depth[b, i, j] = torch.isclose(rendered_depth[b, i, j], sampled_depths, atol=1e-1)

    matching_depth = matching_depth.any(dim=-3)  # (..., v1, h, w)

    mask = in_frustum_mask & non_zero_depth & matching_depth
    return mask

@torch.no_grad()
def calculate_unprojected_mask(views, context_nums):
    '''Calcuate the loss mask for the target views in the batch'''
    target_depth = views["depthmap"][:, context_nums:]
    target_intrinsics = views["camera_intrinsics"][:, context_nums:]
    target_c2w = views["camera_pose"][:, context_nums:]
    context_depth = views["depthmap"][:, :context_nums]
    context_intrinsics = views["camera_intrinsics"][:, :context_nums]
    context_c2w = views["camera_pose"][:, :context_nums]

    target_intrinsics = target_intrinsics[..., :3, :3]
    context_intrinsics = context_intrinsics[..., :3, :3]

    mask = calculate_in_frustum_mask(
        target_depth, target_intrinsics, target_c2w,
        context_depth, context_intrinsics, context_c2w
    )
    return mask

def depth_to_camera_coords(depthmap, camera_intrinsics):
    """
    Convert depth map to 3D camera coordinates.
    
    Args:
        depthmap (BxHxW tensor): Batch of depth maps
        camera_intrinsics (Bx3x3 tensor): Camera intrinsics matrix for each camera
        
    Returns:
        X_cam (BxHxWx3 tensor): 3D points in camera coordinates
        valid_mask (BxHxW tensor): Mask indicating valid depth pixels
    """
    B, H, W = depthmap.shape
    device = depthmap.device
    dtype = depthmap.dtype
    
    # Ensure intrinsics are float
    camera_intrinsics = camera_intrinsics.float()
    
    # Extract focal lengths and principal points
    fx = camera_intrinsics[:, 0, 0]  # (B,)
    fy = camera_intrinsics[:, 1, 1]  # (B,)
    cx = camera_intrinsics[:, 0, 2]  # (B,)
    cy = camera_intrinsics[:, 1, 2]  # (B,)
    
    # Generate pixel grid
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing='ij'
    )
    
    # Reshape for broadcasting: (1, H, W)
    u_grid = u_grid.unsqueeze(0)
    v_grid = v_grid.unsqueeze(0)
    
    # Compute 3D camera coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    z_cam = depthmap  # (B, H, W)
    x_cam = (u_grid - cx.view(B, 1, 1)) * z_cam / fx.view(B, 1, 1)
    y_cam = (v_grid - cy.view(B, 1, 1)) * z_cam / fy.view(B, 1, 1)
    
    # Stack to form (B, H, W, 3)
    X_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    
    # Valid depth mask
    valid_mask = depthmap > 0.0
    
    return X_cam, valid_mask

def depth_to_world_coords_points(
    depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps=1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of depth maps to world coordinates.

    Args:
        depth_map (torch.Tensor): (B, H, W) Depth map
        extrinsic (torch.Tensor): (B, 4, 4) Camera extrinsic matrix (camera-to-world transformation)
        intrinsic (torch.Tensor): (B, 3, 3) Camera intrinsic matrix

    Returns:
        world_coords_points (torch.Tensor): (B, H, W, 3) World coordinates
        camera_points (torch.Tensor): (B, H, W, 3) Camera coordinates
        point_mask (torch.Tensor): (B, H, W) Valid depth mask
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask (B, H, W)
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates (B, H, W, 3)
    camera_points, _ = depth_to_camera_coords(depth_map, intrinsic)

    # Apply extrinsic matrix (camera -> world)
    R_cam_to_world = extrinsic[:, :3, :3]   # (B, 3, 3)
    t_cam_to_world = extrinsic[:, :3, 3]    # (B, 3)

    # Transform (B, H, W, 3) x (B, 3, 3)^T + (B, 3) -> (B, H, W, 3)
    world_coords_points = torch.einsum('bhwi,bji->bhwj', camera_points, R_cam_to_world) + t_cam_to_world[:, None, None, :]

    return world_coords_points, camera_points, point_mask


def closed_form_inverse_se3(se3: torch.Tensor) -> torch.Tensor:
    """
    Efficiently invert batched SE(3) matrices of shape (B, 4, 4).

    Args:
        se3 (torch.Tensor): (B, 4, 4) Transformation matrices

    Returns:
        out (torch.Tensor): (B, 4, 4) Inverse transformation matrices
    """
    assert se3.ndim == 3 and se3.shape[1:] == (4, 4), f"se3 must be (B, 4, 4), got {se3.shape}"
    R = se3[:, :3, :3]        # (B, 3, 3)
    t = se3[:, :3, 3]         # (B, 3)
    Rt = R.transpose(1, 2)    # (B, 3, 3)
    t_inv = -torch.bmm(Rt, t.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    out = se3.new_zeros(se3.shape)
    out[:, :3, :3] = Rt
    out[:, :3, 3] = t_inv
    out[:, 3, 3] = 1.0
    return out

def quat_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def vector_to_camera_matrices(
    cam_vec, image_hw=None, build_intr=True
):
    """Reconstruct extrinsic and intrinsic matrix from vector."""
    # cam_vec: (..., 9)
    intr = None
    # Decompose vector
    t = cam_vec[..., 0:3]
    q = cam_vec[..., 3:7]
    fov_v = cam_vec[..., 7]
    fov_u = cam_vec[..., 8]

    # Build extrinsic: [R|t]
    R = quat_to_rotmat(q)
    ext = torch.cat([R, t.unsqueeze(-1)], dim=-1)

    # Build intrinsic if needed
    if build_intr:
        h, w = image_hw
        fy = h * 0.5 / torch.tan(fov_v * 0.5)
        fx = w * 0.5 / torch.tan(fov_u * 0.5)
        shape = cam_vec.shape[:-1] + (3, 3)
        intr = torch.zeros(shape, device=cam_vec.device, dtype=cam_vec.dtype)
        intr[..., 0, 0] = fx
        intr[..., 1, 1] = fy
        intr[..., 0, 2] = w * 0.5
        intr[..., 1, 2] = h * 0.5
        intr[..., 2, 2] = 1.0

    return ext, intr

import torch
from einops import rearrange

#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
    
def reg_dense_offsets(xyz, shift=6.0):
    d = xyz.norm(dim=-1, keepdim=True)
    return xyz / d.clamp(min=1e-8) * (torch.exp(d - shift) - torch.exp(-shift))

def reg_dense_scales(scales):
    return scales.exp()

def reg_dense_rotation(rotations, eps=1e-8):
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

def reg_dense_sh(sh):
    return rearrange(sh, '... (d_sh xyz) -> ... d_sh xyz', xyz=3)

def reg_dense_opacities(opacities):
    return opacities.sigmoid()

def reg_dense_weights(weights):
    return weights.sigmoid()

class MotionTokenProjector(nn.Module):
    """将 [B, T*V*M, C] 聚合为 [B, M_fixed, C] 的无参模块"""
    def __init__(self, fixed_num_bases=16):
        super().__init__()
        self.fixed_num_bases = fixed_num_bases
    
    def forward(self, x):
        """
        Args:
            x: [B, N, C] where N = T*V*M (variable length)
        
        Returns:
            [B, M_fixed, C] - 聚合后的运动基
        """
        B, N, C = x.shape
        
        # 1. 重塑为 [B, C, N] (通道第一，方便1D池化)
        x = x.permute(0, 2, 1)  # [B, C, N]
        
        # 2. 自适应平均池化: [B, C, N] -> [B, C, M_fixed]
        x_pooled = F.adaptive_avg_pool1d(x, self.fixed_num_bases)
        
        # 3. 重塑回 [B, M_fixed, C]
        x_pooled = x_pooled.permute(0, 2, 1)
        
        return x_pooled  # [B, M_fixed, C]

class Rasterizer:
    def __init__(self, rasterization_mode="classic", packed=True, abs_grad=True, with_eval3d=False,
                 camera_model="pinhole", sparse_grad=False, distributed=False, grad_strategy=DefaultStrategy):
        self.rasterization_mode = rasterization_mode
        self.packed = packed
        self.abs_grad = abs_grad
        self.camera_model = camera_model
        self.sparse_grad = sparse_grad
        self.grad_strategy = grad_strategy
        self.distributed = distributed
        self.with_eval3d = with_eval3d

    def rasterize_splats(
        self,
        means,           # [N, 3] - 单个时间帧的3D点
        quats,           # [N, 4] - 旋转
        scales,          # [N, 3] - 尺度
        opacities,       # [N,] - 透明度
        colors,          # [N, 1, 3] or [N, 3] - SH颜色
        motion_colors,   # [N, 3] - 运动向量
        camtoworlds,     # [S, 4, 4] - 多个相机位姿
        Ks,              # [S, 3, 3] - 多个相机内参
        width: int,
        height: int,
        sh_degree,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # colors, flows, depths, alphas
        total_views = camtoworlds.shape[0]
        all_colors, all_flows, all_depths, all_alphas = [], [], [], []
        
        # 对每个相机视图进行渲染
        for s in range(total_views):
            # 提取单个相机的参数
            cam_pose = camtoworlds[s:s+1]  # [1, 4, 4]
            K = Ks[s:s+1]  # [1, 3, 3]
            
            # 渲染SH颜色
            render_colors, render_alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(cam_pose),  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=width,
                height=height,
                packed=self.packed,
                absgrad=(self.abs_grad if isinstance(self.grad_strategy, DefaultStrategy) else False),
                sparse_grad=self.sparse_grad,
                rasterize_mode=self.rasterization_mode,
                distributed=self.distributed,
                camera_model=self.camera_model,
                with_eval3d=self.with_eval3d,
                render_mode="RGB+ED",
                sh_degree=sh_degree,
            )
            
            # 渲染motion flow
            rendered_flow, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=motion_colors,  # [N, 3] - 运动向量
                viewmats=torch.linalg.inv(cam_pose),  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=width,
                height=height,
                packed=self.packed,
                absgrad=(self.abs_grad if isinstance(self.grad_strategy, DefaultStrategy) else False),
                sparse_grad=self.sparse_grad,
                rasterize_mode=self.rasterization_mode,
                distributed=self.distributed,
                camera_model=self.camera_model,
                with_eval3d=self.with_eval3d,
                render_mode="RGB+ED",
                sh_degree=None,
            )
            
            all_colors.append(render_colors)      # [1, H, W, 4] RGB+depth
            all_flows.append(rendered_flow)       # [1, H, W, 4] flow+depth (但我们只取flow)
            all_depths.append(render_colors[..., 3:])  # [1, H, W, 1] depth from color render
            all_alphas.append(render_alphas)      # [1, H, W, 1] alpha
        
        # 合并所有视图的结果
        all_colors = torch.cat(all_colors, dim=0)  # [S, H, W, 4]
        all_flows = torch.cat(all_flows, dim=0)    # [S, H, W, 4] 
        all_depths = torch.cat(all_depths, dim=0)  # [S, H, W, 1]
        all_alphas = torch.cat(all_alphas, dim=0)  # [S, H, W, 1]
        
        # 分离RGB和depth
        rgb = all_colors[..., :3]  # [S, H, W, 3]
        flow = all_flows[..., :3]  # [S, H, W, 3] - 取flow的前3维
        
        return rgb, flow, all_depths, all_alphas

    def rasterize_batches(self, means, quats, scales, opacities, colors, motion_colors, viewmats, Ks, width, height, sh_degree, T, V):
        """
        Args:
            means: [B*T, N, 3] - B*T个时间帧的3D点
            viewmats: [B, S, 4, 4] - B个batch，S个视图
            colors: [B*T, N, ...] - SH颜色系数
            motion_colors: [B*T, N, 3] - 运动向量
        """
        B, S, _, _ = viewmats.shape  # S = T*V (total views)
        
        rendered_colors, rendered_flows, rendered_depths, rendered_alphas = [], [], [], []
        
        for b in range(B):
            batch_colors, batch_flows, batch_depths, batch_alphas = [], [], [], []
            
            for s in range(S):  # S = T*V views
                # 确定当前视图对应的时间帧
                # 假设视图顺序是 [view0_time0, view1_time0, ..., viewV_time0, view0_time1, ...]
                time_idx = s // V  # s // V gives us the time frame index
                param_idx = b * T + time_idx  # [0*T+0, 0*T+1, ... (B-1)*T+T-1]
                
                # 获取对应时间帧的3D参数
                means_bs = means[param_idx]  # [N, 3]
                quats_bs = quats[param_idx]  # [N, 4] 
                scales_bs = scales[param_idx]  # [N, 3]
                opacities_bs = opacities[param_idx]  # [N,]
                colors_bs = colors[param_idx]  # [N, 1, 3] or [N, 3]
                motion_bs = motion_colors[param_idx]  # [N, 3]
                
                # 当前视图的相机参数
                cam_pose = viewmats[b, s:s+1]  # [1, 4, 4]
                K = Ks[b, s:s+1]  # [1, 3, 3]
                
                # 渲染单个视图
                render_color_s, render_flow_s, render_depth_s, render_alpha_s = self.rasterize_splats(
                    means_bs, quats_bs, scales_bs, opacities_bs, colors_bs, motion_bs,
                    cam_pose, K, width, height, sh_degree
                )
                
                # 移除单视图维度
                batch_colors.append(render_color_s.squeeze(0))    # [H, W, 3]
                batch_flows.append(render_flow_s.squeeze(0))      # [H, W, 3] 
                batch_depths.append(render_depth_s.squeeze(0))    # [H, W, 1]
                batch_alphas.append(render_alpha_s.squeeze(0))    # [H, W, 1]
            
            # 合并batch的所有视图
            rendered_colors.append(torch.stack(batch_colors, dim=0))    # [S, H, W, 3]
            rendered_flows.append(torch.stack(batch_flows, dim=0))      # [S, H, W, 3]
            rendered_depths.append(torch.stack(batch_depths, dim=0))    # [S, H, W, 1] 
            rendered_alphas.append(torch.stack(batch_alphas, dim=0))    # [S, H, W, 1]
        
        # 合并所有batch
        rendered_colors = torch.stack(rendered_colors, dim=0)    # [B, S, H, W, 3]
        rendered_flows = torch.stack(rendered_flows, dim=0)      # [B, S, H, W, 3]
        rendered_depths = torch.stack(rendered_depths, dim=0)    # [B, S, H, W, 1]
        rendered_alphas = torch.stack(rendered_alphas, dim=0)    # [B, S, H, W, 1]
        
        return rendered_colors, rendered_flows, rendered_depths, rendered_alphas
    

class GaussianSplatRenderer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,       # Output channels of gs_feat_head
        sh_degree: int = 0,
        predict_offset: bool = False,
        predict_residual_sh: bool = True,
        enable_prune: bool = False,
        voxel_size: float = 0.002,    # Default voxel size for prune_gs
        using_gtcamera_splat: bool = False,
        render_novel_views: bool = False,
        enable_conf_filter: bool = False,  # Enable confidence filtering
        conf_threshold_percent: float = 30.0,  # Confidence threshold percentage
        max_gaussians: int = 5000000,  # Maximum number of Gaussians
        debug=False,
        num_motion_tokens=16,
    ):
        super().__init__()
        if rasterization is None:
            raise ImportError(
                "GaussianSplatRenderer requires gsplat. Install with: pip install gsplat"
            )

        self.feature_dim = feature_dim
        self.sh_degree = sh_degree              # default: 0
        self.nums_sh = (sh_degree + 1) ** 2     # default: 1
        self.predict_offset = predict_offset
        self.predict_residual_sh = predict_residual_sh
        self.voxel_size = voxel_size
        self.enable_prune = enable_prune
        self.using_gtcamera_splat = using_gtcamera_splat
        self.render_novel_views = render_novel_views
        self.enable_conf_filter = enable_conf_filter
        self.conf_threshold_percent = conf_threshold_percent
        self.max_gaussians = max_gaussians
        self.debug = debug

        # ------- motion predictor -------
        self.num_motion_tokens = num_motion_tokens
        self.tau = 0.5
        num_velocity_channels = 3
        projected_motion_dim = 32
        dim_in = 1024

        # ------- motion predictor -------
        self.motion_key_head = Mlp(128, 256, projected_motion_dim)
        if self.num_motion_tokens > 0:
            self.motion_token_projector = MotionTokenProjector(fixed_num_bases=16)
            self.motion_query_heads = nn.ModuleList(
                [
                    Mlp(dim_in, dim_in, projected_motion_dim)
                    for _ in range(self.num_motion_tokens)
                ]
            )
            self.motion_basis_decoder = Mlp(dim_in, 256, num_velocity_channels)
        else:
            self.motion_tokens = None
            self.motion_basis_decoder = Mlp(projected_motion_dim, 256, num_velocity_channels)

        # Predict Gaussian parameters from GS features (quaternions/scales/opacities/SH/weights/optional offsets)
        if self.predict_offset:
            splits_and_inits = [
                (4, 1.0, 0.0),                # quats
                (3, 0.00003, -2.0),           # scales
                (1, 1.0, -2.0),               # opacities
                (3 * self.nums_sh, 1.0, 0.0), # residual_sh
                (1, 1.0, -2.0),               # weights
                (3, 0.001, 0.001),            # offsets
            ]
            gaussian_raw_channels = 4 + 3 + 1 + self.nums_sh * 3 + 1 + 3
        else:
            splits_and_inits = [
                (4, 1.0, 0.0),                # quats
                (3, 0.00003, -2.0),           # scales
                (1, 1.0, -2.0),               # opacities
                (3 * self.nums_sh, 1.0, 0.0), # residual_sh
                (1, 1.0, -2.0),               # weights
            ]
            gaussian_raw_channels = 4 + 3 + 1 + self.nums_sh * 3 + 1

        self.gs_head = nn.Sequential(
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(feature_dim, gaussian_raw_channels, kernel_size=1),
        )
        # Initialize weights and biases of the final layer by segments
        final_conv_layer = self.gs_head[-1]
        start_channels = 0
        for out_channel, s, b in splits_and_inits:
            nn.init.xavier_uniform_(final_conv_layer.weight[start_channels:start_channels+out_channel], s)
            nn.init.constant_(final_conv_layer.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

        # Rasterizer
        self.rasterizer = Rasterizer()

    # ======== Main entry point: Complete GS rendering and fill results back to predictions ========
    def forward(
        self,
        gs_feats: torch.Tensor,                    # [B, T, V, 3, H, W]
        images: torch.Tensor,                      # [B, T, V, 3, H, W]
        predictions: Dict[str, torch.Tensor],      # From vggt: pose/depth/pts3d etc
        batch: Dict[str, torch.Tensor],
        context_predictions = None,
        render_motion_seg = True,
        motion_tokens = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns predictions with the following fields filled:
        - rendered_colors / rendered_depths / (rendered_alphas during training)
        - gt_colors / gt_depths / valid_masks
        - splats / rendered_extrinsics / rendered_intrinsics
        """
        # 0) init input and target params
        B, T_cur, V , _, H, W = images.shape
        S_cur = T_cur*V
        device = images.device  # ✅ 获取 device

        tar_images = batch['target_images']
        _, T_tar, _ , _, _, _ = tar_images.shape
        S_tar = T_tar*V

        # 2) Select predicted cameras
        time_data = prepare_time_data(batch)
        pred_all_extrinsic, pred_all_intrinsic = self.prepare_prediction_cameras(predictions=predictions, batch=batch, hw=(H, W)) # [B, 2*T, V, 4, 4], [B, 2*T, V, 3, 3]
        pred_all_extrinsic = pred_all_extrinsic.reshape(B, S_cur + S_tar, 4, 4)
        pred_all_intrinsic = pred_all_intrinsic.reshape(B, S_cur + S_tar, 3, 3)

        # 1) Predict GS features from tokens, then convert to Gaussian parameters
        gs_feats_reshape = rearrange(gs_feats, "b s c h w -> (b s) c h w")
        gs_params = self.gs_head(gs_feats_reshape) # btv 12 h w
        
        # 3) Generate splats from gs_params + predictions, and perform voxel merging  -  609168 = 3x2x196x518 - TxVxHxW
        # splats['means'] : torch.Size([2, 609168, 3])
        # splats['quats'] : torch.Size([2, 609168, 4])
        # splats["opacities"] : torch.Size([2, 609168])
        # splats['sh'] : torch.Size([2, 609168, 1, 3])
        # splats['residual_sh'] : torch.Size([2, 609168, 1, 3])
        # splats['weights'] : torch.Size([2, 609168])
        splats = self.prepare_splats(batch, predictions, images, gs_params, S_cur, S_tar, position_from="gsdepth+predcamera", context_predictions=context_predictions, debug=False)
        
        # splats['forward_flow'] : torch.Size([2, 609168])
        splats = self.forward_motion_predictor(gs_feats, T_cur, V, motion_tokens, splats)
        tgt_t = T_cur+T_tar

        # ✅ 警告：这里会复制 tgt_t 倍的高斯点，这是显存占用大的主要原因
        # 例如：60万点 × 6 = 360万点
        splats['means'] = splats['means'].repeat_interleave(tgt_t, dim=0).float()
        splats['scales'] = splats['scales'].repeat_interleave(tgt_t, dim=0).float()
        splats['quats'] = splats['quats'].repeat_interleave(tgt_t, dim=0).float()
        splats['opacities'] = splats['opacities'].repeat_interleave(tgt_t, dim=0).float()
        splats['sh'] = splats['sh'].repeat_interleave(tgt_t, dim=0).float()
        if 'residual_sh' in splats:
            splats['residual_sh'] = splats['residual_sh'].repeat_interleave(tgt_t, dim=0).float()
        splats['weights'] = splats['weights'].repeat_interleave(tgt_t, dim=0).float()
        forward_v = rearrange(splats["forward_flow"], "b t v h w c -> b (t v h w) c")
        splats['forward_flow'] = forward_v.repeat_interleave(tgt_t, dim=0).float()

        ctx_time = time_data["context_time"] * time_data["timespan"][:, None, None]
        tgt_time = interpolate_time(time_data) * time_data["timespan"][:, None, None]

        if tgt_time.ndim == 3:
            tdiff_forward = tgt_time.unsqueeze(2) - ctx_time.unsqueeze(1)
            tdiff_forward = tdiff_forward.view(B * tgt_t, T_tar * V, 1)
            tdiff_forward = tdiff_forward.repeat_interleave(H * W, dim=1).float()
        else:
            tdiff_forward = tgt_time.unsqueeze(-1) - ctx_time.unsqueeze(-2)
            tdiff_forward = tdiff_forward.view(B * tgt_t, T_tar, 1)
            tdiff_forward = tdiff_forward.repeat_interleave(V * H * W, dim=1).float()
        splats['means'] = splats['means'] + splats['forward_flow'] * tdiff_forward

        # Apply confidence filtering before pruning
        # if self.enable_conf_filter and "depth_conf" in predictions:
        #     splats = self.apply_confidence_filter(splats, predictions["depth_conf"])
        if self.enable_prune:
            # splats.keys() : dict_keys(['means', 'sh', 'opacities', 'scales', 'quats'])
            splats = self.prune_gs(splats, voxel_size=self.voxel_size)
        # 4) Rasterization rendering (training: chunked rendering + novel view valid mask correction; evaluation: view-by-view)

        # ✅ 优化：预分配张量，避免列表累积导致的峰值显存
        S_total = S_cur + S_tar

        # 预分配输出张量（使用 float16 节省显存）
        rendered_colors = torch.zeros(B, S_total, H, W, 3, dtype=torch.float16, device=device)
        rendered_depths = torch.zeros(B, S_total, H, W, 1, dtype=torch.float16, device=device)
        rendered_alphas = torch.zeros(B, S_total, H, W, 1, dtype=torch.float16, device=device)
        rendered_flow = torch.zeros(B, S_total, H, W, 3, dtype=torch.float16, device=device)

        chunk_size = V
        for i in range(0, S_total, chunk_size):
            end_idx = min(i + chunk_size, S_total)
            viewmats_i = pred_all_extrinsic[:, i:end_idx] # torch.Size([2, 12, 4, 4])
            Ks_i = pred_all_intrinsic[:, i:end_idx]             # torch.Size([2, 12, 3, 3])

            rendered_colors_chunk, rendered_flow_chunk, rendered_depths_chunk, rendered_alphas_chunk = self.rasterizer.rasterize_batches(
                splats["means"], splats["quats"], splats["scales"], splats["opacities"],
                splats["sh"] if "sh" in splats else splats["colors"], splats["forward_flow"],
                viewmats_i.detach(), Ks_i.detach(),
                width=W, height=H,
                sh_degree=min(self.sh_degree, 0) if "sh" in splats else None,
                T = T_cur+T_tar,
                V = V
            )

            # ✅ 直接写入预分配张量，避免 list 累积
            rendered_colors[:, i:end_idx] = rendered_colors_chunk.to(torch.float16)
            rendered_depths[:, i:end_idx] = rendered_depths_chunk.to(torch.float16)
            rendered_alphas[:, i:end_idx] = rendered_alphas_chunk.to(torch.float16)
            rendered_flow[:, i:end_idx] = rendered_flow_chunk.to(torch.float16)

            # ✅ 及时释放中间变量
            del rendered_colors_chunk, rendered_depths_chunk, rendered_alphas_chunk, rendered_flow_chunk

        # 5) return predictions
        predictions["splats"] = splats

        # torch.Size([2, 6*2, 196, 518, 3])
        output_dict = {
            "rendered_image" : rendered_colors, # imp
            "rendered_depths" : rendered_depths,
            "rendered_alphas" : rendered_alphas,
            "rendered_flow" : rendered_flow  # imp
        }

        # Free large intermediate tensors that are no longer needed after rendering.
        # Keep means/scales/quats/opacities/sh/weights since predictions still reference them.
        if "forward_flow" in splats:
            del splats['forward_flow']
        if "residual_sh" in splats:
            del splats['residual_sh']

        return predictions, output_dict

    def apply_confidence_filter(self, splats, gs_depth_conf):
        """
        Apply confidence filtering to Gaussian splats before pruning.
        Discard bottom p% confidence points, keep top (100-p)%.
        
        Args:
            splats: Dictionary containing Gaussian parameters
            gs_depth_conf: Confidence tensor [B, S, H, W]
        
        Returns:
            Filtered splats dictionary
        """
        if not self.enable_conf_filter or gs_depth_conf is None:
            return splats

        device = splats["means"].device
        _, N = splats["means"].shape[:2]

        # Flatten confidence: [B, S, H, W] -> [B, N]
        conf = gs_depth_conf.flatten(1).to(device)
        # Mask invalid/very small values
        conf = conf.masked_fill(conf <= 1e-5, float("-inf"))

        # Keep top (100-p)% points, discard bottom p%
        if self.conf_threshold_percent > 0:
            keep_from_percent = int(np.ceil(N * (100.0 - self.conf_threshold_percent) / 100.0))
        else:
            keep_from_percent = N
        K = max(1, min(self.max_gaussians, keep_from_percent))

        # Select top-K indices for each batch (deterministic, no randomness)
        topk_idx = torch.topk(conf, K, dim=1, largest=True, sorted=False).indices  # [B, K]
        
        filtered = {}
        mask_keys = ["means", "quats", "scales", "opacities", "sh", "weights"]
        
        for key in splats.keys():
            if key in mask_keys and key in splats:
                x = splats[key]
                if x.ndim == 2:  # [B, N]
                    filtered[key] = torch.gather(x, 1, topk_idx)
                else:
                    # Expand indices to match tensor dimensions
                    expand_idx = topk_idx.clone()
                    for i in range(x.ndim - 2):
                        expand_idx = expand_idx.unsqueeze(-1)
                    expand_idx = expand_idx.expand(-1, -1, *x.shape[2:])
                    filtered[key] = torch.gather(x, 1, expand_idx)
            else:
                filtered[key] = splats[key]

        return filtered

    def prune_gs(self, splats, voxel_size=0.001):
        """
        Prune Gaussian splats by merging those in the same voxel.
        
        Args:
            splats: Dictionary containing Gaussian parameters
            voxel_size: Size of voxels for spatial grouping
            
        Returns:
            Dictionary with pruned splats
        """
        B = splats["means"].shape[0]
        merged_splats_list = []
        device = splats["means"].device

        for i in range(B):
            # Extract splats for current batch
            splats_i = {k: splats[k][i] for k in ["means", "quats", "scales", "opacities", "sh", "weights", "forward_flow"]}
            
            # Compute voxel indices
            coords = splats_i["means"]
            voxel_indices = (coords / voxel_size).floor().long()
            min_indices = voxel_indices.min(dim=0)[0]
            voxel_indices = voxel_indices - min_indices
            max_dims = voxel_indices.max(dim=0)[0] + 1
            
            # Flatten 3D voxel indices to 1D
            flat_indices = (voxel_indices[:, 0] * max_dims[1] * max_dims[2] + 
                           voxel_indices[:, 1] * max_dims[2] + 
                           voxel_indices[:, 2])
            
            # Find unique voxels and inverse mapping
            unique_voxels, inverse_indices = torch.unique(flat_indices, return_inverse=True)
            K = len(unique_voxels)

            # Initialize merged splats
            merged = {
                "means": torch.zeros((K, 3), device=device),
                "quats": torch.zeros((K, 4), device=device),
                "scales": torch.zeros((K, 3), device=device),
                "opacities": torch.zeros(K, device=device),
                "sh": torch.zeros((K, self.nums_sh, 3), device=device),
                "forward_flow": torch.zeros((K, 3), device=device)
            }
            
            # Get weights and compute weight sums per voxel
            weights = splats_i["weights"]
            weight_sums = torch.zeros(K, device=device)
            weight_sums.scatter_add_(0, inverse_indices, weights)
            weight_sums = torch.clamp(weight_sums, min=1e-8)

            # Merge means (weighted average)
            for d in range(3):
                merged["means"][:, d].scatter_add_(0, inverse_indices, 
                                                 splats_i["means"][:, d] * weights)
            merged["means"] = merged["means"] / weight_sums.unsqueeze(1)

            # Merge spherical harmonics (weighted average)
            for d in range(3):
                merged["sh"][:, 0, d].scatter_add_(0, inverse_indices, 
                                                  splats_i["sh"][:, 0, d] * weights)
            merged["sh"] = merged["sh"] / weight_sums.unsqueeze(-1).unsqueeze(-1)

            # Merge opacities (weighted sum of squares)
            merged["opacities"].scatter_add_(0, inverse_indices, weights * weights)
            merged["opacities"] = merged["opacities"] / weight_sums

            # Merge scales (weighted average)
            for d in range(3):
                merged["scales"][:, d].scatter_add_(0, inverse_indices, 
                                                  splats_i["scales"][:, d] * weights)
            merged["scales"] = merged["scales"] / weight_sums.unsqueeze(1)

            # Merge quaternions (weighted average + normalization)
            for d in range(4):
                merged["quats"][:, d].scatter_add_(0, inverse_indices, 
                                                 splats_i["quats"][:, d] * weights)
            quat_norms = torch.norm(merged["quats"], dim=1, keepdim=True)
            merged["quats"] = merged["quats"] / torch.clamp(quat_norms, min=1e-8)

            # Merge means (weighted average)
            for d in range(3):
                merged["forward_flow"][:, d].scatter_add_(0, inverse_indices, 
                                                 splats_i["forward_flow"][:, d] * weights)
            merged["forward_flow"] = merged["forward_flow"] / weight_sums.unsqueeze(1)

            merged_splats_list.append(merged)

        # Reorganize output
        output = {}
        for key in ["means", "sh", "opacities", "scales", "quats", "forward_flow"]:
            output[key] = [merged[key] for merged in merged_splats_list]
        
        return output

    def prepare_splats(self, batch, predictions, images, gs_params, context_nums, target_nums, context_predictions=None, position_from="gsdepth+predcamera", debug=False):
        """
        Prepare Gaussian splats from model predictions and input data.
        
        Args:
            views: Dictionary containing view data (camera poses, intrinsics, etc.)
            predictions: Model predictions including depth, pose_enc, etc.
            images: Input images [B, S_all, 3, H, W]
            gs_params: Gaussian splatting parameters from model
            context_nums: Number of context views (S)
            target_nums: Number of target views (V)
            context_predictions: Optional context predictions for camera poses
            position_from: Method to compute 3D positions ("pts3d", "preddepth+predcamera", "gsdepth+predcamera", "gsdepth+gtcamera")
            debug: Whether to use debug mode with ground truth data
            
        Returns:
            splats: Dictionary containing prepared Gaussian splat parameters
        """
        B, T, V, C, H, W  = images.shape
        S_cur, S_tar = context_nums, target_nums
        images = images.view(B, T*V, C, H, W)
        splats = {}
        
        # Only take parameters from source view branch
        gs_params = rearrange(gs_params, "(b s) c h w -> b s h w c", b=B)[:, :S_cur]
        splats["gs_feats"] = gs_params.reshape(B, S_cur*H*W, -1)

        # Split Gaussian parameters based on whether offset prediction is enabled
        if self.predict_offset:
            quats, scales, opacities, residual_sh, weights, offsets = torch.split(
                gs_params, [4, 3, 1, self.nums_sh * 3, 1, 3], dim=-1
            )
            offsets = reg_dense_offsets(offsets.reshape(B, S * H * W, 3))
            splats["offsets"] = offsets
        else:
            quats, scales, opacities, residual_sh, weights = torch.split(
                gs_params, [4, 3, 1, self.nums_sh * 3, 1], dim=-1
            )
            offsets = 0.

        # Apply activation functions to Gaussian parameters
        splats["quats"] = reg_dense_rotation(quats.reshape(B, S_cur * H * W, 4))
        splats["scales"] = reg_dense_scales(scales.reshape(B, S_cur * H * W, 3)).clamp(0.001, 0.1)
        splats["opacities"] = reg_dense_opacities(opacities.reshape(B, S_cur * H * W))
        residual_sh = reg_dense_sh(residual_sh.reshape(B, S_cur * H * W, self.nums_sh * 3))

        # Handle spherical harmonics (SH) coefficients
        if self.predict_residual_sh:
            new_sh = torch.zeros_like(residual_sh)
            new_sh[..., 0, :] = RGB2SH(
                images[:, :S_cur].permute(0, 1, 3, 4, 2).reshape(B, S_cur * H * W, 3)
            )
            splats['sh'] = new_sh + residual_sh
            splats['residual_sh'] = residual_sh
        else:
            splats['sh'] = residual_sh

        splats["weights"] = reg_dense_weights(weights.reshape(B, S_cur * H * W))

        # Compute 3D positions based on specified method

        # current point
        if position_from == "world_points":
            pts3d = predictions["world_points"][:, :S_cur].reshape(B, S_cur * H * W, 3)
            splats["means"] = pts3d + offsets
            
        elif position_from == "preddepth+predcamera":
            depth = predictions["depth"][:, :S_cur].reshape(B * S_cur, H, W)
            if context_predictions is not None:
                pose3x4, intrinsic = vector_to_camera_matrices(
                    context_predictions["camera_params"][:, :S_cur].reshape(B * S_cur, -1), (H, W)
                )
            else:
                pose3x4, intrinsic = vector_to_camera_matrices(
                    predictions["pose_enc"][:, :S_cur].reshape(B * S_cur, -1), (H, W)
                )
            pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * S, 1, 1)
            pose4x4[:, :3, :4] = pose3x4
            extrinsics = closed_form_inverse_se3(pose4x4)
            pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
            pts3d = pts3d.reshape(B, S_cur * H * W, 3)
            splats["means"] = pts3d + offsets
            
        elif position_from == "gsdepth+predcamera":
            depth = predictions["gs_depth"][:, :S_cur].reshape(B * S_cur, H, W)
            if context_predictions is not None:
                pose3x4, intrinsic = vector_to_camera_matrices(
                    context_predictions["camera_params"][:, :S_cur].reshape(B * S_cur, -1), (H, W)
                )
            else:
                pose3x4, intrinsic = vector_to_camera_matrices(
                    predictions["pose_enc"][:, :S_cur].reshape(B * S_cur, -1), (H, W)
                )
            pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * S_cur, 1, 1)
            pose4x4[:, :3, :4] = pose3x4
            extrinsics = closed_form_inverse_se3(pose4x4)
            pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
            pts3d = pts3d.reshape(B, S_cur * H * W, 3)
            splats["means"] = pts3d + offsets
            
        # elif position_from == "gsdepth+gtcamera":
        #     depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
        #     pose4x4 = batch["input_extrinsic"][:, :S].reshape(B * S, 4, 4)
        #     intrinsic = batch["input_intrinsics"][:, :S].reshape(B * S, 3, 3)
        #     extrinsics = pose4x4
        #     pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
        #     pts3d = pts3d.reshape(B, S * H * W, 3)
        #     splats["means"] = pts3d + offsets
            
        else:
            raise ValueError(f"Invalid position_from={position_from}")

        return splats

    def forward_motion_predictor(self, gs_feats, t, v, motion_tokens=None, splats=None):
        b, s, _, h, w = gs_feats.shape # torch.Size([2, 6, 128, 196, 518])
        
        gs_feats = rearrange(gs_feats, "b (t v) c h w -> b t v h w c", t=t, v=v)
        img_keys = self.motion_key_head(gs_feats)

        if self.num_motion_tokens > 0 and motion_tokens is not None:
            # === 关键: 从 [B, T, V, M, C] 转换为 [B, K, C] ===
            B, T, V, M, C = motion_tokens.shape
            motion_flat = motion_tokens.reshape(B, T * V * M, C)
            K = T * V * M  # 总 token 数, 运动基元
            motion_flat = self.motion_token_projector(motion_flat) # (B, M, C)

            hyper_in_list = []
            for i in range(self.num_motion_tokens):
                hyper_in = self.motion_query_heads[i](motion_flat[:, i])
                hyper_in_list.append(hyper_in)
            motion_token_queries = torch.stack(hyper_in_list, dim=1)
            motion_bases = self.motion_basis_decoder(motion_flat)
            dot_product_similarity = torch.einsum(
                "b k c, b t v h w c -> b t v h w k",
                motion_token_queries,
                img_keys,
            )
            motion_weights = torch.softmax(dot_product_similarity / self.tau, dim=-1)
            forward_flow = torch.einsum("b t v h w k, b k c -> b t v h w c", motion_weights, motion_bases)
            splats["motion_weights"] = motion_weights
            splats["motion_bases"] = motion_bases
        else:
            # if there's no motion token, directly predict the velocity from the upsampled image features
            forward_flow = self.motion_basis_decoder(img_keys) # torch.Size([B, T, V, H, W, 3])

        splats["forward_flow"] = forward_flow
        return splats

    def prepare_cameras(self, views, nums):
        viewmats = views['camera_pose'][:, :nums]
        Ks = views['camera_intrinsics'][:, :nums]
        return viewmats, Ks

    def prepare_prediction_cameras(self, predictions, batch, hw: Tuple[int, int], ):
        """
        Prepare camera matrices from predicted pose encodings.
        
        Args:
            predictions: Dictionary containing pose_enc predictions
            nums: Number of views to process
            hw: Tuple of (height, width)
            
        Returns:
            viewmats: Camera view matrices [B, S, 4, 4]
            Ks: Camera intrinsic matrices [B, S, 3, 3]
        """
        B, T, V, C, H, W = batch["input_images"].shape
        S = T*V
        H, W = hw

        tar_pose3x4 = batch['target_extrinsics']
        
        # Convert pose encoding to extrinsics and intrinsics
        pose3x4, intrinsic = vector_to_camera_matrices(
            predictions["pose_enc"][:, :S].reshape(B * S, -1), (H, W)
        )

        # Convert to homogeneous coordinates and compute view matrices
        pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * S, 1, 1)
        pose4x4[:, :3, :4] = pose3x4

        viewmats = closed_form_inverse_se3(pose4x4).reshape(B, T, V, 4, 4)
        Ks = intrinsic.reshape(B, T, V, 3, 3)
        
        # interpolate for render pose
        viewmats = interpolate_extrinsics(viewmats)                     # [B, 2*T, V, 4, 4]
        Ks = interpolate_intrinsics(Ks, batch['target_intrinsics'])     # [B, 2*T, V, 3, 3]

        return viewmats, Ks
            
def interpolate_extrinsics(predicted_extrinsics, target_poses=None):
    """
    插值中间帧并外推最后一帧
    
    Args:
        predicted_extrinsics: [B, T, V, 4, 4] 预测的0,2,4...帧外参 (虚拟坐标系)
        target_poses: [B, M, V, 4, 4] 真实的1,3,5...帧外参 (真实坐标系)，用于计算运动关系
                     M 通常是 T (如果预测0,2则target是1,3; 如果预测0,2,4则target是1,3,5)
    
    Returns:
        result: [B, 2*T, V, 4, 4] 完整的0,1,2,3,4,5...帧外参 (真实坐标系)
    """
    B, T, V, _, _ = predicted_extrinsics.shape
    device = predicted_extrinsics.device
    dtype = predicted_extrinsics.dtype
    
    total_frames = 2 * T  # 0,1,2,3,4,5... (T个偶数帧 + T个奇数帧)
    result = torch.zeros(B, total_frames, V, 4, 4, device=device, dtype=dtype)
    
    for b in range(B):
        for v in range(V):
            pred_poses = predicted_extrinsics[b, :, v]  # [T, 4, 4]
            true_target_poses = target_poses[b, :, v] if target_poses is not None else None  # [T, 4, 4]
            
            # 构建完整序列
            full_sequence = _interpolate_and_extrapolate_with_motion(pred_poses, true_target_poses, total_frames)
            result[b, :, v] = full_sequence
    
    return result

def _interpolate_and_extrapolate_with_motion(pred_poses, true_target_poses, total_frames):
    """
    插值中间帧，使用真实帧运动关系外推最后一帧
    """
    T = pred_poses.shape[0]  # 预测帧数
    
    # 初始化结果
    result = torch.eye(4, 4, device=pred_poses.device, dtype=pred_poses.dtype).unsqueeze(0).repeat(total_frames, 1, 1)
    
    # 填入预测的偶数帧 (0,2,4 -> 索引0,2,4)
    for i in range(T):
        even_idx = i * 2  # 0->0, 1->2, 2->4, ...
        if even_idx < total_frames:
            result[even_idx] = pred_poses[i]
    
    # 插值奇数帧 (1,3,5 -> 索引1,3,5) - 这些也是虚拟坐标系
    for i in range(T):
        even_idx = i * 2      # 0,2,4
        odd_idx = even_idx + 1 # 1,3,5
        
        if odd_idx < total_frames:
            next_even_idx = (i + 1) * 2
            if next_even_idx < total_frames:
                # 在相邻偶数帧之间插值奇数帧 (虚拟坐标系内插值)
                result[odd_idx] = _interpolate_pose_se3(result[even_idx], result[next_even_idx], 0.5)
            else:
                # 如果没有下一个偶数帧，使用运动外推
                if i > 0:
                    prev_even_idx = (i - 1) * 2
                    # 使用前一个运动趋势外推
                    prev_motion = torch.matmul(result[even_idx], torch.inverse(result[prev_even_idx]))
                    result[odd_idx] = torch.matmul(prev_motion, result[even_idx])
    
    # 如果有真实目标帧，用它们来计算运动关系并外推最后一帧
    if true_target_poses is not None and true_target_poses.shape[0] >= 2:
        # 计算真实帧间的运动关系，用于外推
        # 比如：如果真实帧是[1,3]，计算从1到3的运动；如果真实帧是[1,3,5]，计算从3到5的运动趋势
        
        last_target_idx = total_frames - 1  # 要外推的帧索引 (3 或 5)
        
        if last_target_idx == 3 and true_target_poses.shape[0] >= 2:  # 0,1,2,3 情况
            # 使用真实帧1->3的运动关系
            true_1_pose = true_target_poses[0]  # 真实帧1
            true_3_pose = true_target_poses[1]  # 真实帧3
            
            # 计算真实1->3的运动
            true_motion_1_to_3 = torch.matmul(true_3_pose, torch.inverse(true_1_pose))
            
            # 应用坐标系对齐
            pred_1_pose = result[1]  # 预测插值的帧1
            alignment_transform = torch.matmul(true_1_pose, torch.inverse(pred_1_pose))
            
            # 对齐所有预测帧
            aligned_result = result.clone()
            for idx in range(total_frames):
                if not torch.allclose(aligned_result[idx], torch.zeros(4, 4, device=pred_poses.device)):
                    aligned_result[idx] = torch.matmul(alignment_transform, aligned_result[idx])
            
            # 现在aligned_result[1]应该等于true_1_pose
            # 我们用真实运动关系来设置帧3
            aligned_result[3] = torch.matmul(true_motion_1_to_3, aligned_result[1])
            
            # 更新结果
            result = aligned_result
            
        elif last_target_idx >= 5 and true_target_poses.shape[0] >= 3:  # 0,1,2,3,4,5 情况
            # 使用真实帧3->5的运动关系 (或1->3, 3->5的运动趋势)
            true_1_pose = true_target_poses[0]  # 真实帧1
            true_3_pose = true_target_poses[1]  # 真实帧3  
            true_5_pose = true_target_poses[2]  # 真实帧5
            
            # 计算坐标系对齐
            pred_1_pose = result[1]
            alignment_transform = torch.matmul(true_1_pose, torch.inverse(pred_1_pose))
            
            # 对齐所有预测帧
            aligned_result = result.clone()
            for idx in range(total_frames):
                if not torch.allclose(aligned_result[idx], torch.zeros(4, 4, device=pred_poses.device)):
                    aligned_result[idx] = torch.matmul(alignment_transform, aligned_result[idx])
            
            # 使用真实运动来校正帧3
            aligned_result[3] = torch.matmul(true_3_pose, torch.inverse(true_1_pose)) @ aligned_result[1]
            
            # 外推帧5：使用真实3->5的运动
            aligned_result[5] = torch.matmul(true_5_pose, torch.inverse(true_3_pose)) @ aligned_result[3]
            
            result = aligned_result
    
    return result

def _interpolate_pose_se3(pose1, pose2, t=0.5):
    """
    在SE(3)空间中插值两个变换矩阵
    """
    import torch
    
    # 分离旋转和平移
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]
    
    # 平移线性插值
    t_interp = (1 - t) * t1 + t * t2
    
    # 旋转SLERP插值 (简化版本)
    R_rel = torch.matmul(R2, R1.transpose(-2, -1))
    
    # 计算旋转角度
    trace = torch.trace(R_rel)
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    if theta.abs() < 1e-6:
        R_interp = (1 - t) * R1 + t * R2
        R_interp = R_interp / (torch.det(R_interp).abs().sqrt().reshape(-1, 1, 1))  # 保证行列式为1
    else:
        # 使用轴角表示进行插值
        sin_theta = torch.sin(theta)
        K = (R_rel - R_rel.transpose(-2, -1)) / (2 * sin_theta + 1e-8)
        
        # Rodrigues公式
        R_t = (torch.eye(3, device=R1.device, dtype=R1.dtype) + 
               torch.sin(t * theta) * K + 
               (1 - torch.cos(t * theta)) * torch.matmul(K, K))
        
        R_interp = torch.matmul(R_t, R1)
    
    # 组合结果
    result = torch.eye(4, device=pose1.device, dtype=pose1.dtype)
    result[:3, :3] = R_interp
    result[:3, 3] = t_interp
    
    return result
        
if __name__ == "__main__":
    device = "cuda:0"
    means = torch.randn((100, 3), device=device)
    quats = torch.randn((100, 4), device=device)
    scales = torch.rand((100, 3), device=device) * 0.1  
    opacities = torch.rand((100,), device=device)
    colors = torch.rand((100, 3), device=device)

    viewmats = torch.eye(4, device=device)[None, :, :].repeat(10, 1, 1)
    Ks = torch.tensor([
    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :].repeat(10, 1, 1)
    width, height = 300, 200

    rasterizer = Rasterizer()
    splats = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }
    colors, alphas, _ = rasterizer.rasterize_splats(splats, viewmats, Ks, width, height)
    
