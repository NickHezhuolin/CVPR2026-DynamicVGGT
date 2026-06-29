# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil, floor
from typing import Union, Optional
from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from train_utils.general import check_and_fix_inf_nan
import math
from math import ceil, floor
import utils3d

from train_utils.alignment import (
    align_points_scale_z_shift, 
    align_points_scale, 
    align_points_scale_xyz_shift,
    align_points_z_shift,
)
from train_utils.geometry_torch import (
    weighted_mean, 
    harmonic_mean, 
    geometric_mean,
    mask_aware_nearest_resize,
    normalized_view_plane_uv,
    angle_diff_vec3
)

def fast_cross(a, b):
    result = torch.empty_like(a)
    result[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    result[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    result[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return result

@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.

    Supports:
    - Camera loss
    - Depth loss
    - Point loss
    - Future point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """

    def __init__(self, camera=None, depth=None, point=None, future_point=None, track=None, is_multiframe=True, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.future_point = future_point
        self.track = track
        self.is_multiframe = is_multiframe

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.

        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}

        # Camera pose loss - if pose encodings are predicted
        ##compute_depth_loss compute_camera_loss
        if "pose_enc_list" in predictions:
            # camera_loss_dict = camera_loss(predictions, batch, **self.camera)
            camera_loss_dict = compute_camera_loss(predictions["pose_enc_list"], batch, "huber")
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)

        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions["depth"], predictions["depth_conf"], batch,
                                                 gradient_loss="grad")

            # depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]

            if self.is_multiframe:
                depth_loss = depth_loss_dict["loss_conf1_depth"] + depth_loss_dict["loss_reg1_depth"] + depth_loss_dict[
                    "loss_grad1_depth"] + depth_loss_dict["loss_conf2_depth"] + depth_loss_dict["loss_reg2_depth"] + \
                             depth_loss_dict["loss_grad2_depth"]
            else:
                depth_loss = depth_loss_dict["loss_conf1_depth"] + depth_loss_dict["loss_reg1_depth"] + depth_loss_dict[
                    "loss_grad1_depth"]
                depth_loss_dict.pop("loss_conf2_depth")
                depth_loss_dict.pop("loss_reg2_depth")
                depth_loss_dict.pop("loss_grad2_depth")

            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)
        
        # 3D point reconstruction loss - if world points are predicted
        if "world_points" in predictions and self.point is not None:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict[
                "loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # Future 3D point loss - supervised against target_world_points
        if "future_world_points" in predictions and self.future_point is not None:
            future_point_loss_dict = compute_future_point_loss(predictions, batch, **self.future_point)
            future_point_loss = (
                future_point_loss_dict["loss_conf_future_point"]
                + future_point_loss_dict["loss_reg_future_point"]
                + future_point_loss_dict["loss_grad_future_point"]
            )
            future_point_loss = future_point_loss * self.future_point["weight"]
            total_loss = total_loss + future_point_loss
            loss_dict.update(future_point_loss_dict)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            coord_preds, vis_scores, conf_scores = predictions["track"], predictions["vis"], predictions["conf"]
            track_loss_dict = compute_track_loss(coord_preds, vis_scores, conf_scores, batch)
            track_loss = track_loss_dict["loss_track"] + track_loss_dict["loss_track_vis"] + track_loss_dict["loss_track_conf"]
            track_loss = track_loss * self.track["weight"]
            total_loss = total_loss + track_loss
            loss_dict.update(track_loss_dict)

        loss_dict["loss_total"] = total_loss

        return loss_dict

# hzl - edit
def closed_form_scale_and_shift(pred, gt, valid_mask):
    """ 
    Args:
        pred:   (B, S, H, W, C) 
        gt:     (B, S, H, W, C) 
        valid_mask: (B, S, H, W) 
    Returns:
        scale:  (B, S) 
        shift:  (B, S, C) 
    """
    assert pred.dim() == 5 and gt.dim() == 5 and valid_mask.dim() == 4, "Inputs must be 5D tensors"
    B, S, H, W, C = pred.shape
    device = pred.device

    if C == 1: # 需要切换到 B,1 的输出
        # 扩展valid_mask以匹配pred的维度
        valid_mask_expanded = valid_mask.unsqueeze(-1).expand(B, S, H, W, C).reshape(B, -1, C)
        valid_mask_flat = valid_mask_expanded.any(dim=-1)  # (B, N) 任何通道有效就算有效

        pred_flat = pred.reshape(B, -1, C)  # (B, N, C)
        gt_flat = gt.reshape(B, -1, C)      # (B, N, C)

        scales = []
        shifts = []
        for i in range(B):
            # 获取当前batch的有效点
            batch_valid = valid_mask_flat[i]  # (N,)
            
            if batch_valid.sum() < 10:  # 如果有效点太少，使用默认值
                scales.append(torch.tensor(1.0, device=device))
                shifts.append(torch.zeros(C, device=device))
                continue
                
            pred_valid = pred_flat[i][batch_valid]  # (M, C)
            gt_valid = gt_flat[i][batch_valid]      # (M, C)
            
            # 单通道：独立计算每个batch的尺度和偏移
            pred_mean = pred_valid.mean(dim=0)
            gt_mean = gt_valid.mean(dim=0)
            
            numerator = ((pred_valid - pred_mean) * (gt_valid - gt_mean)).sum(dim=0)
            denominator = ((pred_valid - pred_mean) ** 2).sum(dim=0).clamp(min=1e-6)
            scale = numerator / denominator
            shift = gt_mean - scale * pred_mean

            scales.append(scale.mean() if scale.dim() > 0 else scale) 
            shifts.append(shift)

        scale_tensor = torch.stack(scales).reshape(B, S)  # (B, S)
        shift_tensor = torch.stack(shifts).reshape(B, S, C)  # (B, S, C)
    elif C == 3:
        (pred_points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize((pred, gt), mask=valid_mask, size=(32, 32))
        scale_tensor, shift_tensor = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2),lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=1.0)
    else:
        raise ValueError(f"Unsupported channel dimension C={C}")
    
    return scale_tensor, shift_tensor

def compute_track_loss(coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds.shape[2] ### shape was not right, maybe the input was track_list
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=1000)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)

    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        # print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0

        loss_dict = {
            f"loss_track": dummy_coord,
            f"loss_track_vis": dummy_vis,
            f"loss_track_conf": dummy_conf,
        }
        return loss_dict

    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=[coord_preds],
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())
    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=100)

    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=100)
    else:
        conf_loss = 0

    loss_dict = {
        f"loss_track": track_loss,
        f"loss_track_vis": vis_loss,
        f"loss_track_conf": conf_loss,
    }
    return loss_dict

def sequence_loss(flow_preds, flow_gt, vis, valids, 
                  gamma=0.8, vis_aware=False, huber=False, delta=10, 
                  vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=1000)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=1000)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean

def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """

    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


def compute_future_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1, **kwargs):
    """
    Compute future point loss against target-world-point supervision.

    Args:
        predictions: Dict with 'future_world_points' and 'future_world_points_conf'
        batch: Dict with 'target_world_points' and 'target_point_masks'
    """
    pred_points = predictions["future_world_points"]
    pred_points_conf = predictions["future_world_points_conf"]
    gt_points = batch["target_world_points"]
    gt_points_mask = batch["target_point_masks"]

    if gt_points.dim() == 6:
        b, t, v, h, w, c = gt_points.shape
        gt_points = gt_points.reshape(b, t * v, h, w, c)
        gt_points_mask = gt_points_mask.reshape(b, t * v, h, w)

    seq_len = min(pred_points.shape[1], gt_points.shape[1])
    if seq_len < pred_points.shape[1]:
        pred_points = pred_points[:, :seq_len]
        pred_points_conf = pred_points_conf[:, :seq_len]
    if seq_len < gt_points.shape[1]:
        gt_points = gt_points[:, :seq_len]
        gt_points_mask = gt_points_mask[:, :seq_len]

    point_loss_dict = compute_point_loss(
        {"world_points": pred_points, "world_points_conf": pred_points_conf},
        {"world_points": gt_points, "point_masks": gt_points_mask},
        gamma=gamma,
        alpha=alpha,
        gradient_loss_fn=gradient_loss_fn,
        valid_range=valid_range,
        **kwargs,
    )
    return {
        "loss_conf_future_point": point_loss_dict["loss_conf_point"],
        "loss_reg_future_point": point_loss_dict["loss_reg_point"],
        "loss_grad_future_point": point_loss_dict["loss_grad_point"],
    }


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1, **kwargs):
    """
    Compute point loss.

    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']

    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")

    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                     f"loss_reg_point": dummy_loss,
                     f"loss_grad_point": dummy_loss, }
        return loss_dict

    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                                     gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha,
                                                     valid_range=valid_range)

    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }

    return loss_dict


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.

    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss

    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering

    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")

    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb * ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb * ss, hh, ww, nc),
            gt.reshape(bb * ss, hh, ww, nc),
            mask.reshape(bb * ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb * ss, hh, ww, nc),
            gt.reshape(bb * ss, hh, ww, nc),
            mask.reshape(bb * ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range > 0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range > 0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn=None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = F.huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3])
        loss_R = F.huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
        loss_fl = F.huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100)  # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


def compute_camera_loss(pred_pose_enc_list, batch, loss_type="l1", gamma=0.6, pose_encoding_type="absT_quaR_FoV",
                        weight_T=1.0,
                        weight_R=1.0, weight_fl=0.5, frame_num=-100):
    # Extract predicted and ground truth components
    mask_valid = batch['point_masks']

    batch_valid_mask = mask_valid[:, 0].sum(dim=[-1, -2]) > 100
    num_predictions = len(pred_pose_enc_list)

    gt_extrinsic = batch['extrinsics']
    gt_intrinsic = batch['intrinsics']
    image_size_hw = batch['images'].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(gt_extrinsic, gt_intrinsic, image_size_hw,
                                                    pose_encoding_type=pose_encoding_type)

    loss_T = loss_R = loss_fl = 0

    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)

        cur_pred_pose_enc = pred_pose_enc_list[i]

        if batch_valid_mask.sum() == 0:
            loss_T_i = (cur_pred_pose_enc * 0).mean()
            loss_R_i = (cur_pred_pose_enc * 0).mean()
            loss_fl_i = (cur_pred_pose_enc * 0).mean()
        else:
            if frame_num > 0:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(
                    cur_pred_pose_enc[batch_valid_mask][:, :frame_num].clone(),
                    gt_pose_encoding[batch_valid_mask][:, :frame_num].clone(), loss_type=loss_type)
            else:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc[batch_valid_mask].clone(),
                                                                   gt_pose_encoding[batch_valid_mask].clone(),
                                                                   loss_type=loss_type)
        loss_T += loss_T_i * i_weight
        loss_R += loss_R_i * i_weight
        loss_fl += loss_fl_i * i_weight

    loss_T = loss_T / num_predictions
    loss_R = loss_R / num_predictions
    loss_fl = loss_fl / num_predictions
    loss_camera = loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    loss_dict = {
        "loss_camera": loss_camera,
        "loss_T": loss_T,
        "loss_R": loss_R,
        "loss_fl": loss_fl
    }

    return loss_dict


def compute_depth_loss(depth, depth_conf, batch, gamma=1.0, alpha=0.2, loss_type="conf", predict_disparity=False,
                       affine_inv=False, gradient_loss=None, valid_range=-1, disable_conf=False, all_mean=False,
                       single_frame=True, **kwargs):
    gt_depth = batch['depths'].clone()
    valid_mask = batch['point_masks']
    valid_mask = valid_mask.bool()

    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")

    gt_depth = gt_depth[..., None]
    # gt_depth_mask = batch['point_masks'].clone()  # 3D points derived from depth map, so we use the same mask
    # if gt_depth_mask.sum() < 100:
    #     # If there are less than 100 valid points, skip this batch
    #     dummy_loss = (0.0 * depth).mean()
    #     loss_dict = {f"loss_conf1_depth": dummy_loss,
    #                  f"loss_reg1_depth": dummy_loss,
    #                  f"loss_grad1_depth": dummy_loss,
    #                  f"loss_conf2_depth": dummy_loss,
    #                  f"loss_reg2_depth": dummy_loss,
    #                  f"loss_grad2_depth": dummy_loss,
    #                  }
    #     return loss_dict

    if loss_type == "conf":
        conf_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask,
                                   batch, normalize_pred=False, normalize_gt=False,
                                   gamma=gamma, alpha=alpha, affine_inv=affine_inv, gradient_loss=gradient_loss,
                                   valid_range=valid_range, postfix="_depth", disable_conf=disable_conf,
                                   all_mean=all_mean, single_frame=single_frame)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return conf_loss_dict


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filters a loss tensor by keeping only values below a certain quantile threshold.
    Also clamps individual values to hard_max.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= 1000:
        # too small, just return
        return loss_tensor

    # Randomly sample if tensor is too large
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values
    loss_tensor = loss_tensor.clamp(max=hard_max)

    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask, batch, normalize_gt=True, normalize_pred=True, gamma=1.0,
              alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1,
              disable_conf=False, all_mean=False, postfix="", single_frame=True):
    # normalize
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    if affine_inv:
        scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        pts3d = pts3d * scale + shift

    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames = reg_loss(pts3d,
                                                                                                          gt_pts3d,
                                                                                                          valid_mask,
                                                                                                          gradient_loss=gradient_loss)

    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        conf_loss_first_frame = gamma * loss_reg_first_frame * first_frame_conf[first_frame_mask] - alpha * torch.log(
            first_frame_conf[first_frame_mask])
        conf_loss_other_frames = gamma * loss_reg_other_frames * other_frames_conf[
            other_frames_mask] - alpha * torch.log(other_frames_conf[other_frames_mask])

    single_frame = pts3d_conf.shape[1] == 1
    if conf_loss_first_frame.numel() > 0 and (single_frame or conf_loss_other_frames.numel() > 0):
        if valid_range > 0:
            conf_loss_first_frame = filter_by_quantile(conf_loss_first_frame, valid_range)
            conf_loss_other_frames = filter_by_quantile(conf_loss_other_frames, valid_range)

        conf_loss_first_frame = check_and_fix_inf_nan(conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
        conf_loss_other_frames = check_and_fix_inf_nan(conf_loss_other_frames, f"conf_loss_other_frames{postfix}")
    else:
        conf_loss_first_frame = pts3d * 0
        conf_loss_other_frames = pts3d * 0
        print("No valid conf loss", batch["seq_name"])

    if all_mean and conf_loss_first_frame.numel() > 0 and (single_frame or conf_loss_other_frames.numel() > 0):
        all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])

        # for logging only
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0

    # Verified that the loss is the same

    loss_dict = {
        f"loss_reg1{postfix}": loss_reg_first_frame.mean() if loss_reg_first_frame.numel() > 0 else 0,
        f"loss_reg2{postfix}": loss_reg_other_frames.mean() if loss_reg_other_frames.numel() > 0 else 0,
        f"loss_conf1{postfix}": conf_loss_first_frame,
        f"loss_conf2{postfix}": conf_loss_other_frames,
    }

    if gradient_loss is not None:
        # loss_grad_first_frame and loss_grad_other_frames are already meaned
        loss_dict[f"loss_grad1{postfix}"] = loss_grad_first_frame
        loss_dict[f"loss_grad2{postfix}"] = loss_grad_other_frames

    return loss_dict

# hzl - edit
def normalize_pointcloud(pts3d, valid_mask, eps=1e-3, normalize_per_frame=True):
    """
    Normalize pointcloud with support for both 4D and 5D inputs.
    
    Args:
        pts3d: B, H, W, 3 或 B, S, H, W, 3
        valid_mask: B, H, W 或 B, S, H, W
        eps: 避免除零的小值
        normalize_per_frame: 如果为True, 对每个帧独立归一化; 如果为False, 对整个序列归一化
    
    Returns:
        normalized_pts3d: 归一化后的点云，形状与输入相同
        avg_scale: 归一化尺度，形状为 (B,) 或 (B, S)
    """
    # 检查输入维度
    if pts3d.dim() == 4:
        # 原始4D输入：B, H, W, 3
        dist = pts3d.norm(dim=-1)  # (B, H, W)
        dist_sum = (dist * valid_mask).sum(dim=[1, 2])  # (B,)
        valid_count = valid_mask.sum(dim=[1, 2])  # (B,)
        
        avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)  # (B,)
        pts3d = pts3d / avg_scale.view(-1, 1, 1, 1)  # 广播到 (B, H, W, 3)
        
    elif pts3d.dim() == 5:
        # 新的5D输入：B, S, H, W, 3
        B, S, H, W, C = pts3d.shape
        
        if normalize_per_frame:
            # 对每个帧独立归一化
            dist = pts3d.norm(dim=-1)  # (B, S, H, W)
            dist_sum = (dist * valid_mask).sum(dim=[2, 3])  # (B, S)
            valid_count = valid_mask.sum(dim=[2, 3])  # (B, S)
            
            avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)  # (B, S)
            pts3d = pts3d / avg_scale.view(B, S, 1, 1, 1)  # 广播到 (B, S, H, W, 3)
            
        else:
            # 对整个序列归一化（合并S维度）
            # 将点云和掩码重塑为4D
            pts3d_flat = pts3d.reshape(B, S*H, W, C)  # (B, S*H, W, 3)
            valid_mask_flat = valid_mask.reshape(B, S*H, W)  # (B, S*H, W)
            
            dist = pts3d_flat.norm(dim=-1)  # (B, S*H, W)
            dist_sum = (dist * valid_mask_flat).sum(dim=[1, 2])  # (B,)
            valid_count = valid_mask_flat.sum(dim=[1, 2])  # (B,)
            
            avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)  # (B,)
            pts3d = pts3d / avg_scale.view(B, 1, 1, 1, 1)  # 广播到 (B, S, H, W, 3)
            
    else:
        raise ValueError(f"Unsupported input dimension: {pts3d.dim()}. Expected 4 or 5 dimensions.")
    
    return pts3d, avg_scale

def reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=None):
    first_frame_pts3d = pts3d[:, 0:1, ...]
    first_frame_gt_pts3d = gt_pts3d[:, 0:1, ...]
    first_frame_mask = valid_mask[:, 0:1, ...]

    other_frames_pts3d = pts3d[:, 1:, ...]
    other_frames_gt_pts3d = gt_pts3d[:, 1:, ...]
    other_frames_mask = valid_mask[:, 1:, ...]

    loss_reg_first_frame = torch.norm(first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask],
                                      dim=-1)
    loss_reg_other_frames = torch.norm(other_frames_gt_pts3d[other_frames_mask] - other_frames_pts3d[other_frames_mask],
                                       dim=-1)

    if gradient_loss == "grad":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww))
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww))
    elif gradient_loss == "grad_impl2":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww),
                                                          gradient_loss_fn=gradient_loss_impl2)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww),
                                                           gradient_loss_fn=gradient_loss_impl2)
    elif gradient_loss == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww),
                                                          gradient_loss_fn=normal_loss, scales=3)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww),
                                                           gradient_loss_fn=normal_loss, scales=3)
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0

    loss_reg_first_frame = check_and_fix_inf_nan(loss_reg_first_frame, "loss_reg_first_frame")
    loss_reg_other_frames = check_and_fix_inf_nan(loss_reg_other_frames, "loss_reg_other_frames")

    return loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None):
    """
    Computes the normal-based loss by comparing the angle between
    predicted normals and ground-truth normals.

    prediction: (B, H, W, 3) - Predicted 3D coordinates/points
    target:     (B, H, W, 3) - Ground-truth 3D coordinates/points
    mask:       (B, H, W)    - Valid pixel mask (1 = valid, 0 = invalid)

    Returns: scalar (averaged over valid regions)
    """
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)
    del pred_valids, gt_valids 

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    # pred_normals and gt_normals are (4, B, H, W, 3)
    # We want to compare corresponding normals where all_valid is True
    dot = torch.sum(pred_normals * gt_normals, dim=-1)  # shape: (4, B, H, W)
    del pred_normals, gt_normals
    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot  # shape: (4, B, H, W)

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()
            del all_valid
            gamma = 1.0  # hard coded
            alpha = 0.2  # hard coded

            loss = gamma * loss * conf - alpha * torch.log(conf)
            del conf
            return loss.mean()
        else:
            return loss.mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='constant', value=0).permute(0, 2, 3, 1)
        
        del point_map, mask
        # Each pixel's neighbors
        center = pts[:, 1:-1, 1:-1, :]  # B,H,W,3
        up = pts[:, :-2, 1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:, 1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        # Direction vectors
        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        del up, left, down, right
        # Four cross products (shape: B,H,W,3 each)
        # start_events[4].record()
        # n10 = torch.cross(up_dir, left_dir, dim=-1)  # up x left
        # n20 = torch.cross(left_dir, down_dir, dim=-1)  # left x down
        # n30 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        # n40 = torch.cross(right_dir, up_dir, dim=-1)  # right x up
        n1 = fast_cross(up_dir, left_dir)
        n2 = fast_cross(left_dir, down_dir)
        n3 = fast_cross(down_dir, right_dir)
        n4 = fast_cross(right_dir, up_dir)

        del up_dir, left_dir, down_dir, right_dir
        # Validity for each cross-product direction
        # We require that both directions' pixels are valid
        v1 = padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1]
        v3 = padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1]

        del padded_mask
        # Stack them to shape (4,B,H,W,3), (4,B,H,W)
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        del n1, n2, n3, n4, v1, v2, v3, v4
        # Normalize each direction's normal
        # shape is (4, B, H, W, 3), so dim=-1 is the vector dimension
        # clamp_min(eps) to avoid division by zero
        # lengths = torch.norm(normals, dim=-1, keepdim=True).clamp_min(eps)
        # normals = normals / lengths
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1)

    return normals, valids


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        gamma = 1.0
        alpha = 0.2

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss


def gradient_loss_multi_scale(prediction, target, mask, scales=4, gradient_loss_fn=gradient_loss, conf=None):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


# def torch_quantile(
#     input: torch.Tensor,
#     q: float | torch.Tensor,
#     dim: int | None = None,
#     keepdim: bool = False,
#     *,
#     interpolation: str = "nearest",
#     out: torch.Tensor | None = None,
def torch_quantile(
        input: torch.Tensor,
        q: Union[float, torch.Tensor],
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        interpolation: str = "nearest",
        out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''
