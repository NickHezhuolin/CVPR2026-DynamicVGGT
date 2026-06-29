from abc import ABC

import bisect
import random
from typing import List, Tuple, Any
from torch.utils.data import Dataset
from hydra.utils import instantiate
import torch
import numpy as np

from torch.utils.data import ConcatDataset
from .dataset_util import *
from .track_util import *
from .augmentation import get_image_augmentation

class MultiComposedDataset(Dataset, ABC):
    """
    安全的多数据集组合，使用 MyTupleConcatDataset 确保正确的索引路由
    完全兼容原始 ComposedDataset 的所有功能
    """
    def __init__(self, dataset_configs: List[dict], common_config: dict, **kwargs):
        base_dataset_list = []
        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use your custom concatenation class that supports tuple indexing
        self.base_dataset = MyTupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = common_config.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = common_config.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = None
        # get_image_augmentation(
        #     color_jitter=common_config.augs.color_jitter,
        #     gray_scale=common_config.augs.gray_scale,
        #     gau_blur=common_config.augs.gau_blur,
        # )

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- DynamicVGGT settings ---
        self.multiview_and_temporal = common_config.multiview_and_temporal

        # --- Track Settings ---
        # Whether to include point tracks in the output
        self.load_track = common_config.load_track
        # Number of point tracks to include per sequence
        self.track_num = common_config.track_num

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = common_config.training
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples

    def __get_test_item(self, idx_tuple):
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]

        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)
        sample = {
            "images": images
        }
        return sample

    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares tracks if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        # if not self.training:
        #     return self.__get_test_item(idx_tuple)
        if self.multiview_and_temporal:
            if self.fixed_num_images > 0:
                seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
                idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)
            
            # Retrieve the raw data batch from the appropriate base dataset
            batch = self.base_dataset[idx_tuple]
            seq_name = batch["seq_name"]
            
            # --- Data Conversion and Preparation for Multi-view Temporal ---
            # Process input data
            input_data = batch["input"]
            target_data = batch["target"]
            
            # Convert input images to tensors
            input_images = torch.from_numpy(input_data["images"].astype(np.float32)).contiguous()
            input_images = input_images.permute(0, 1, 4, 2, 3).to(torch.get_default_dtype()).div(255)  # (T, V, 3, H, W)
            
            # Convert input other data to tensors
            input_depths = torch.from_numpy(input_data["depths"].astype(np.float32))  # (T, V, H, W)
            input_extrinsics = torch.from_numpy(input_data["extrinsics"].astype(np.float32))  # (T, V, 3, 4)
            input_intrinsics = torch.from_numpy(input_data["intrinsics"].astype(np.float32))  # (T, V, 3, 3)
            input_cam_points = torch.from_numpy(np.stack(input_data["cam_points"]).astype(np.float32))  # (T, V, ...)
            input_world_points = torch.from_numpy(np.stack(input_data["world_points"]).astype(np.float32))  # (T, V, ...)
            input_point_masks = torch.from_numpy(np.stack(input_data["point_masks"]))  # (T, V, ...)
            
            # Convert target data to tensors
            target_images = torch.from_numpy(target_data["images"].astype(np.float32)).contiguous()
            target_images = target_images.permute(0, 1, 4, 2, 3).to(torch.get_default_dtype()).div(255)  # (T, V, 3, H, W)
            
            target_depths = torch.from_numpy(target_data["depths"].astype(np.float32))  # (T, V, H, W)
            target_extrinsics = torch.from_numpy(target_data["extrinsics"].astype(np.float32))  # (T, V, 3, 4)
            target_intrinsics = torch.from_numpy(target_data["intrinsics"].astype(np.float32))  # (T, V, 3, 3)
            target_cam_points = torch.from_numpy(np.stack(target_data["cam_points"]).astype(np.float32))  # (T, V, ...)
            target_world_points = torch.from_numpy(np.stack(target_data["world_points"]).astype(np.float32))  # (T, V, ...)
            target_point_masks = torch.from_numpy(np.stack(target_data["point_masks"]))  # (T, V, ...)
            
            # Process ego-motion data
            ego_motion = batch["ego_motion"]
            relative_poses = torch.from_numpy(ego_motion["relative_poses"].astype(np.float32))  # (T-1, V, 4, 4)
            translations = torch.from_numpy(ego_motion["translations"].astype(np.float32))  # (T-1, V, 3)
            speeds = torch.from_numpy(ego_motion["speeds"].astype(np.float32))  # (T-1, V)
            cumulative_displacement = torch.from_numpy(ego_motion["cumulative_displacement"].astype(np.float32))  # (T, V, 3)
            world_positions = torch.from_numpy(ego_motion["world_positions"].astype(np.float32))  # (T, V, 3)
            
            # Invalidate all points if first frame has no valid points (for input)
            if input_point_masks.numel() > 0 and input_point_masks[0].sum() == 0:
                input_point_masks[:] = False
            
            # Invalidate all points if first frame has no valid points (for target)
            if target_point_masks.numel() > 0 and target_point_masks[0].sum() == 0:
                target_point_masks[:] = False
            
            # --- Apply Color Augmentation (training mode only) ---
            if self.training and self.image_aug is not None:
                if self.cojitter and random.random() > self.cojitter_ratio:
                    # Apply the same color jittering transformation to all frames and views
                    T, V = input_images.shape[:2]
                    # Reshape to (T*V, 3, H, W) for augmentation
                    input_images_flat = input_images.view(-1, *input_images.shape[2:])
                    input_images_flat = self.image_aug(input_images_flat)
                    input_images = input_images_flat.view(T, V, *input_images.shape[2:])
                    
                    target_images_flat = target_images.view(-1, *target_images.shape[2:])
                    target_images_flat = self.image_aug(target_images_flat)
                    target_images = target_images_flat.view(T, V, *target_images.shape[2:])
                else:
                    # Apply different color jittering to each frame-view combination
                    for t in range(input_images.shape[0]):
                        for v in range(input_images.shape[1]):
                            input_images[t, v] = self.image_aug(input_images[t, v])
                            target_images[t, v] = self.image_aug(target_images[t, v])
            
            # --- Prepare Final Sample Dictionary ---
            sample = {
                "seq_name": seq_name,
                "input_ids": torch.tensor(batch["input_ids"]),
                "target_ids": torch.tensor(batch["target_ids"]),
                "stride": batch["stride"],
                "time_gap": batch["time_gap"],
                "frame_num": batch["frame_num"],
                "view_num": batch["view_num"],
                
                # Input data
                "input_images": input_images,  # (T, V, 3, H, W)
                "input_depths": input_depths,  # (T, V, H, W)
                "input_extrinsics": input_extrinsics,  # (T, V, 3, 4)
                "input_intrinsics": input_intrinsics,  # (T, V, 3, 3)
                "input_cam_points": input_cam_points,
                "input_world_points": input_world_points,
                "input_point_masks": input_point_masks,
                
                # Target data
                "target_images": target_images,  # (T, V, 3, H, W)
                "target_depths": target_depths,  # (T, V, H, W)
                "target_extrinsics": target_extrinsics,  # (T, V, 3, 4)
                "target_intrinsics": target_intrinsics,  # (T, V, 3, 3)
                "target_cam_points": target_cam_points,
                "target_world_points": target_world_points,
                "target_point_masks": target_point_masks,
                
                # Ego-motion data
                "relative_poses": relative_poses,  # (T-1, V, 4, 4)
                "translations": translations,  # (T-1, V, 3)
                "speeds": speeds,  # (T-1, V)
                "cumulative_displacement": cumulative_displacement,  # (T, V, 3)
                "world_positions": world_positions,  # (T, V, 3)
            }

            # process flows ans sky_masks
            has_flow = (
                input_data["flows"] is not None 
                and target_data["flows"] is not None
            )

            has_sky_mask = (
                input_data["sky_masks"] is not None 
                and target_data["sky_masks"] is not None 
            )

            has_moge_depth = (
                input_data["moge_depths"] is not None 
                and target_data["moge_depths"] is not None 
            )

            sample.update({
                "has_flow": has_flow,
                "has_sky_mask": has_sky_mask,
                "has_moge_depth": has_moge_depth
            })

            if has_flow:
                input_flows = torch.from_numpy(np.stack(input_data["flows"])) # (T, V, H, W, 3)
                target_flows = torch.from_numpy(np.stack(target_data["flows"])) # (T, V, H, W, 3)
                sample.update({
                    "input_flows": input_flows,
                    "target_flows": target_flows,
                })

            if has_sky_mask:
                input_sky_masks = torch.from_numpy(np.stack(input_data["sky_masks"]))
                target_sky_masks = torch.from_numpy(np.stack(target_data["sky_masks"]))
                sample.update({
                    "input_sky_masks": input_sky_masks,
                    "target_sky_masks": target_sky_masks,
                })
            
            if has_moge_depth:
                input_moge_depths = torch.from_numpy(np.stack(input_data["moge_depths"]))
                input_moge_cam_points = torch.from_numpy(np.stack(input_data["moge_cam_points"]).astype(np.float32))
                input_moge_world_points = torch.from_numpy(np.stack(input_data["moge_world_points"]).astype(np.float32))
                target_moge_depths = torch.from_numpy(np.stack(target_data["moge_depths"]))
                target_moge_cam_points = torch.from_numpy(np.stack(target_data["moge_cam_points"]).astype(np.float32))
                target_moge_world_points = torch.from_numpy(np.stack(target_data["moge_world_points"]).astype(np.float32))
                sample.update({
                    "input_moge_depths": input_moge_depths,
                    "input_moge_cam_points": input_moge_cam_points,
                    "input_moge_world_points": input_moge_world_points,

                    "target_moge_depths": target_moge_depths,
                    "target_moge_cam_points": target_moge_cam_points,
                    "target_moge_world_points": target_moge_world_points,
                })

        # else:
        #     # single view - video
        #     # If fixed settings are provided, override the tuple values
        #     if self.fixed_num_images > 0:
        #         seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
        #         idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        #     # Retrieve the raw data batch from the appropriate base dataset
        #     batch = self.base_dataset[idx_tuple]
        #     seq_name = batch["seq_name"]

        #     # --- Data Conversion and Preparation ---
        #     # Convert numpy arrays to tensors
        #     images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        #     # Normalize images from [0, 255] to [0, 1]
        #     images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)

        #     # Convert other data to tensors with appropriate types
        #     depths = torch.from_numpy(np.stack(batch["depths"]).astype(np.float32))
        #     extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))
        #     intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
        #     cam_points = torch.from_numpy(np.stack(batch["cam_points"]).astype(np.float32))
        #     world_points = torch.from_numpy(np.stack(batch["world_points"]).astype(np.float32))
        #     point_masks = torch.from_numpy(np.stack(batch["point_masks"])) # Mask indicating valid depths / world points / cam points per frame
        #     ids = torch.from_numpy(batch["ids"])    # Frame indices sampled from the original sequence


        #     # Invalidate all points if first frame has no valid points
        #     if point_masks.numel() > 0 and point_masks[0].sum() == 0:
        #         point_masks[:] = False

        #     # --- Apply Color Augmentation (training mode only) ---
        #     if self.training and self.image_aug is not None:
        #         if self.cojitter and random.random() > self.cojitter_ratio:
        #             # Apply the same color jittering transformation to all frames
        #             images = self.image_aug(images)
        #         else:
        #             # Apply different color jittering to each frame individually
        #             for aug_img_idx in range(len(images)):
        #                 images[aug_img_idx] = self.image_aug(images[aug_img_idx])


        #     # --- Prepare Final Sample Dictionary ---
        #     sample = {
        #         "seq_name": seq_name,
        #         "ids": ids,
        #         "images": images,
        #         "depths": depths,
        #         "extrinsics": extrinsics,
        #         "intrinsics": intrinsics,
        #         "cam_points": cam_points,
        #         "world_points": world_points,
        #         "point_masks": point_masks,
        #     }

        # # --- Track Processing (if enabled) ---
        # if self.load_track:
        #     # import time
        #     # start = time.perf_counter()
        #     if "tracks" in batch and batch["tracks"] is not None:
        #         # Use pre-computed tracks from the dataset
        #         tracks = torch.from_numpy(np.stack(batch["tracks"]).astype(np.float32))
        #         track_vis_mask = torch.from_numpy(np.stack(batch["track_masks"]).astype(bool))

        #         # Sample a subset of tracks randomly
        #         valid_indices = torch.where(track_vis_mask[0])[0]
        #         if len(valid_indices) >= self.track_num:
        #             # If we have enough tracks, sample without replacement
        #             sampled_indices = valid_indices[torch.randperm(len(valid_indices))][:self.track_num]
        #         else:
        #             # If not enough tracks, sample with replacement (allow duplicates)
        #             sampled_indices = valid_indices[torch.randint(0, len(valid_indices),
        #                                             (self.track_num,),
        #                                             dtype=torch.int64,
        #                                             device=valid_indices.device)]

        #         # Extract the sampled tracks and their masks
        #         tracks = tracks[:, sampled_indices, :]
        #         track_vis_mask = track_vis_mask[:, sampled_indices]
        #         track_positive_mask = torch.ones(track_vis_mask.shape[1]).bool()

        #     else:
        #         # Generate tracks on-the-fly using depth information
        #         # This creates synthetic tracks based on the 3D information available
        #         tracks, track_vis_mask, track_positive_mask = build_tracks_by_depth(
        #             extrinsics, intrinsics, world_points, depths, point_masks, images,
        #             target_track_num=self.track_num, seq_name=seq_name
        #         )

        #     # Add track information to the sample dictionary
        #     sample["tracks"] = tracks
        #     sample["track_vis_mask"] = track_vis_mask
        #     sample["track_positive_mask"] = track_positive_mask
        #     # print("time for tracks:", time.perf_counter() - start)
        return sample


class MyTupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple with safe index to diffirent datasets.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (datasets_idx, sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample in a datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across a datasets, not all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = False

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # print(f"DEBUG: MyTupleConcatDataset.__getitem__ - global_idx={idx}, dataset_idx={dataset_idx}, sample_idx={sample_idx}")

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        result = self.datasets[dataset_idx][idx_tuple]
        
        # 添加数据集信息用于验证
        if isinstance(result, dict):
            result['debug_dataset_idx'] = dataset_idx
            result['debug_global_idx'] = idx
            
        return result
