import bisect
import random
import numpy as np
import torch
from typing import Callable, Optional, List
from torch.utils.data import DataLoader, Dataset, Sampler
from hydra.utils import instantiate
from .worker_fn import get_worker_init_fn
from abc import ABC, abstractmethod

class MultiDatasetDynamicTorchDataset(ABC):
    """
    多数据集版本的 DynamicTorchDataset
    """
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
        max_img_per_gpu: int = 48,
        sampling_strategy: str = 'round_robin'
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu
        self.sampling_strategy = sampling_strategy

        # Instantiate the dataset (this will be ComposedDataset)
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = common_config.augs.aspects
        self.image_num_range = common_config.img_nums

        # Validate the aspect ratio and image number ranges
        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # 不使用 DynamicDistributedSampler
        self.sampler = None
        
        # 获取 cumulative_sizes 从 ComposedDataset
        cumulative_sizes = None
        if hasattr(self.dataset, 'base_dataset') and hasattr(self.dataset.base_dataset, 'cumulative_sizes'):
            cumulative_sizes = self.dataset.base_dataset.cumulative_sizes
            print(f"Successfully got cumulative_sizes: {cumulative_sizes}")
        else:
            print("WARNING: Could not get cumulative_sizes, falling back to single dataset mode")
        
        self.batch_sampler = MultiDatasetDynamicBatchSampler(
            dataset=self.dataset,
            aspect_ratio_range=self.aspect_ratio_range,
            image_num_range=self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu,
            cumulative_sizes=cumulative_sizes,
            sampling_strategy=sampling_strategy
        )

    def get_loader(self, epoch):
        print(f"Building multi-dataset dataloader with epoch: {epoch}")

        self.batch_sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )


class MultiDatasetDynamicBatchSampler(Sampler):
    """
    确保每个 batch 都来自同一个底层数据集
    """
    def __init__(self,
                 dataset,  # 直接接收 dataset
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48,
                 cumulative_sizes=None,
                 sampling_strategy='round_robin'):
        self.dataset = dataset
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random(seed)
        self.max_img_per_gpu = max_img_per_gpu
        self.sampling_strategy = sampling_strategy
        self.cumulative_sizes = cumulative_sizes or []
        
        # 计算每个数据集的长度
        if self.cumulative_sizes:
            self.dataset_lengths = []
            for i in range(len(self.cumulative_sizes)):
                if i == 0:
                    self.dataset_lengths.append(self.cumulative_sizes[i])
                else:
                    self.dataset_lengths.append(self.cumulative_sizes[i] - self.cumulative_sizes[i-1])
            self.num_datasets = len(self.dataset_lengths)
            self.sampling_order = list(range(self.num_datasets))
            self.current_dataset_idx = 0
        else:
            self.dataset_lengths = []
            self.num_datasets = 0
        
        # 初始化采样参数
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)
        
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng.seed(epoch * 100)
        
        if self.cumulative_sizes and self.sampling_strategy == 'round_robin':
            self.rng.shuffle(self.sampling_order)
            self.current_dataset_idx = 0

    def _get_dataset_idx(self, global_idx):
        """根据全局索引确定属于哪个数据集"""
        if not self.cumulative_sizes:
            return 0
        
        if global_idx >= self.cumulative_sizes[-1]:
            return len(self.cumulative_sizes) - 1
        
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, global_idx)
        return dataset_idx

    def __iter__(self):
        if not self.cumulative_sizes:
            # 退化为简单随机采样
            total_samples = len(self.dataset)
            indices = list(range(total_samples))
            self.rng.shuffle(indices)
            
            # 生成 batch
            i = 0
            while i < len(indices):
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)
                batch_size = max(1, self.max_img_per_gpu // random_image_num)
                
                current_batch = []
                for _ in range(batch_size):
                    if i >= len(indices):
                        break
                    current_batch.append((indices[i], random_image_num, random_aspect_ratio))
                    i += 1
                
                if current_batch:
                    yield current_batch
        else:
            # 按数据集分组索引
            # 按数据集分组索引
            dataset_indices = [[] for _ in range(self.num_datasets)]
            total_samples = self.cumulative_sizes[-1]
            
            # print(f"DEBUG: total_samples = {total_samples}")
            # print(f"DEBUG: cumulative_sizes = {self.cumulative_sizes}")
            # print(f"DEBUG: num_datasets = {self.num_datasets}")
            
            for global_idx in range(total_samples):
                dataset_idx = self._get_dataset_idx(global_idx)
                if dataset_idx < self.num_datasets:
                    dataset_indices[dataset_idx].append(global_idx)
            
            # 打印每个数据集的索引范围
            # for i, indices in enumerate(dataset_indices):
            #     if indices:
            #         print(f"DEBUG: Dataset {i}: {len(indices)} samples, range [{min(indices)}, {max(indices)}]")
            
            # 打乱每个数据集内部的索引
            for indices in dataset_indices:
                self.rng.shuffle(indices)
            
            # 创建迭代器
            dataset_iterators = [iter(indices) for indices in dataset_indices]
            active_datasets = set(range(self.num_datasets))
            
            while active_datasets:
                # 选择数据集
                if self.sampling_strategy == 'round_robin':
                    dataset_idx = self.sampling_order[self.current_dataset_idx % len(self.sampling_order)]
                    self.current_dataset_idx += 1
                elif self.sampling_strategy == 'random':
                    dataset_idx = self.rng.choice(list(active_datasets))
                else:
                    dataset_idx = 0
                
                if dataset_idx not in active_datasets:
                    continue
                    
                dataset_iter = dataset_iterators[dataset_idx]
                
                # 采样动态参数
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)
                
                # 计算 batch size
                batch_size = max(1, self.max_img_per_gpu // random_image_num)
                batch_size = int(batch_size)

                # print(f"DEBUG: Selected dataset {dataset_idx} for batch")
                
                # 从当前数据集收集 batch
                current_batch = []
                collected_datasets = []
                for _ in range(batch_size):
                    try:
                        global_idx = next(dataset_iter)
                        current_batch.append((global_idx, random_image_num, random_aspect_ratio))
                        actual_dataset = self._get_dataset_idx(global_idx)
                        collected_datasets.append(actual_dataset)
                        # print(f"  DEBUG: global_idx={global_idx} -> dataset={actual_dataset}")
                    except StopIteration:
                        break
                
                # print(f"DEBUG: Batch datasets: {collected_datasets}")
                if len(set(collected_datasets)) > 1:
                    print("ERROR: Mixed datasets in batch!")
                    raise RuntimeError("Mixed datasets detected!")
                
                if current_batch:
                    yield current_batch
                else:
                    active_datasets.discard(dataset_idx)

    def __len__(self):
        total_samples = self.cumulative_sizes[-1]
        return total_samples


def debug_collate_fn(batch):
    """
    调试用的 collate 函数，详细检查形状不一致问题
    """
    if not batch:
        return {}
    
    print(f"\n{'='*50}")
    print(f"DEBUG COLLATE: Processing batch of size {len(batch)}")
    print(f"{'='*50}")

    # 检查每个样本的 debug 信息
    for i, sample in enumerate(batch):
        if 'debug_dataset_idx' in sample:
            print(f"  Sample {i}: dataset_idx={sample['debug_dataset_idx']}, global_idx={sample['debug_global_idx']}")
        if 'dataset_info' in sample:
            print(f"  Sample {i}: dataset_info={sample['dataset_info']}")
    
    # 获取第一个样本作为参考
    first_sample = batch[0]
    
    def check_and_collate(key, values):
        """检查并尝试 collate 一个字段"""
        print(f"\nChecking key: '{key}'")
        
        # 检查类型一致性
        types = [type(v) for v in values]
        if len(set(types)) > 1:
            print(f"  ❌ Type mismatch: {[t.__name__ for t in types]}")
            return values  # 返回原列表
        
        first_val = values[0]
        print(f"  First value type: {type(first_val)}")
        
        if isinstance(first_val, torch.Tensor):
            # 检查张量形状
            shapes = [v.shape for v in values]
            print(f"  Shapes: {shapes}")
            
            if len(set(shapes)) == 1:
                print(f"  ✅ All shapes match: {shapes[0]}")
                try:
                    return torch.stack(values, 0)
                except Exception as e:
                    print(f"  ❌ Stack failed: {e}")
                    return values
            else:
                print(f"  ❌ Shape mismatch! Unique shapes: {set(shapes)}")
                # 找出具体哪些样本形状不同
                for i, shape in enumerate(shapes):
                    print(f"    Sample {i}: {shape}")
                return values  # 返回原列表
                
        elif isinstance(first_val, np.ndarray):
            print(f"  NumPy array shape: {first_val.shape}")
            shapes = [v.shape for v in values]
            if len(set(shapes)) == 1:
                try:
                    return torch.stack([torch.from_numpy(v) for v in values], 0)
                except Exception as e:
                    print(f"  ❌ NumPy stack failed: {e}")
                    return values
            else:
                print(f"  ❌ NumPy shape mismatch: {set(shapes)}")
                return values
                
        elif isinstance(first_val, (list, tuple)):
            print(f"  List/tuple length: {len(first_val)}")
            # 对于列表，递归检查（如果是张量列表）
            if len(first_val) > 0 and isinstance(first_val[0], torch.Tensor):
                # 检查列表中每个张量的形状
                for i, item_list in enumerate(values):
                    if len(item_list) != len(first_val):
                        print(f"  ❌ List length mismatch at sample {i}: {len(item_list)} vs {len(first_val)}")
                        break
                else:
                    print(f"  ✅ All list lengths match")
            return values
            
        else:
            print(f"  Non-tensor type: {type(first_val)}")
            return values
    
    # 处理字典结构
    if isinstance(first_sample, dict):
        result = {}
        for key in first_sample.keys():
            values = [sample[key] for sample in batch]
            result[key] = check_and_collate(key, values)
        return result
    
    # 处理列表结构
    elif isinstance(first_sample, (list, tuple)):
        result = []
        for i in range(len(first_sample)):
            values = [sample[i] for sample in batch]
            result.append(check_and_collate(f"index_{i}", values))
        return type(first_sample)(result)
    
    else:
        print(f"Unexpected batch type: {type(first_sample)}")
        return batch
