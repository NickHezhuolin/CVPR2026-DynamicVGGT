# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import os.path as osp
import sys

# sys.path.append(os.dirname(__file__) + '../')
# import pdb
# pdb.set_trace()
import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import cv2
import matplotlib.cm as cm
from sklearn.linear_model import RANSACRegressor
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch, normalize_camera_extrinsics_and_points_batch_for_multiview_and_temporal, normalize_flow_batch_for_multiview_and_temporal
from train_utils.optimizer import construct_optimizers
# from metric_func_multinode import *

# hzl - edit
import utils3d

class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
            self,
            *,
            data: Dict[str, Any],
            model: Dict[str, Any],
            logging: Dict[str, Any],
            checkpoint: Dict[str, Any],
            max_epochs: int,
            mode: str = "train",
            device: str = "cuda",
            seed_value: int = 123,
            val_epoch_freq: int = 1,
            val_freq_while_training: int = -1,
            save_freq_while_training: int = -1,
            distributed: Dict[str, bool] = None,
            cuda: Dict[str, bool] = None,
            limit_train_batches: Optional[int] = None,
            limit_val_batches: Optional[int] = None,
            optim: Optional[Dict[str, Any]] = None,
            loss: Optional[Dict[str, Any]] = None,
            env_variables: Optional[Dict[str, Any]] = None,
            accum_steps: int = 1,
            scale_by_points: bool = True,
            is_multiframe: bool = True,
            use_ransac: bool = False,
            start_epoch_value: int = -1,
            **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()
        # abs and rel
        self.scale_by_points = scale_by_points

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.val_freq_while_training = val_freq_while_training
        self.save_freq_while_training = save_freq_while_training

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        self.is_multiframe = is_multiframe
        self.use_ransac = use_ransac
        self.start_data_iter = 0

        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0
        self.start_epoch_value = start_epoch_value

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # hzl: edit
        self._lpips_enabled = False

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.load_pretrained_path:
            self._load_pretrain_vggt(self.checkpoint_conf.load_pretrained_path)
        elif self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)

        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_pretrain_vggt(self, ckpt_path: str):
        if self.rank == 0:
            logging.info(f"Training from pretrain model: {ckpt_path} (rank {self.rank})")

        # load model
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)

        if self.rank == 0:
            logging.info(
                f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")
        self._init_future_point_head_from_point_head()

    def _init_future_point_head_from_point_head(self):
        """Re-initialize future_point_head from point_head after loading pretrained weights."""
        model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        if hasattr(model, "init_future_point_head_from_point_head"):
            model.init_future_point_head_from_point_head()
            if self.rank == 0:
                logging.info("Initialized future_point_head from point_head weights.")

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(
                f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        if any("future_point_head" in key for key in missing):
            self._init_future_point_head_from_point_head()

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")

            if isinstance(self.optims, list):
                self.optims[0].optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                self.optims.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        if self.start_epoch_value != -1:
            self.epoch = self.start_epoch_value

        if "data_iter" in checkpoint:
            self.start_data_iter = checkpoint["data_iter"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # 新增：读取 LPIPS 启用配置
        self.lpips_start_epoch = getattr(self.loss_conf, 'lpips_start_epoch', 5)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_open_dataset = None
        self.train_own_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:

            if self.data_conf.get('train_open_dataset', None) is None:
                self.train_open_dataset = None
            else:
                self.train_open_dataset = instantiate(self.data_conf.train_open_dataset, _recursive_=False)
                self.train_open_dataset.seed = self.seed_value

            if self.data_conf.get('train_own_dataset', None) is None:
                if self.data_conf.get('train', None) is not None:
                    self.train_own_dataset = instantiate(self.data_conf.train, _recursive_=False)
                    self.train_own_dataset.seed = self.seed_value
                else:
                    self.train_own_dataset = None
            else:
                self.train_own_dataset = instantiate(self.data_conf.train_own_dataset, _recursive_=False)
                self.train_own_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, data_iter: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                    self.checkpoint_conf.save_freq > 0
                    and int(epoch) % self.checkpoint_conf.save_freq == 0
                    and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "data_iter": data_iter,
            "epoch": epoch,
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }

        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models=None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            # self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)

            if self.train_open_dataset is None:
                train_open_dataloader = []
            else:
                train_open_dataloader = self.train_open_dataset.get_loader(
                    epoch=int(self.epoch + self.distributed_rank))

            if self.train_own_dataset is None:
                train_own_dataloader = []
            else:
                train_own_dataloader = self.train_own_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))

            self.train_epoch(train_open_dataloader, train_own_dataloader)

            # Clean up memory
            del train_open_dataloader
            del train_own_dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            # if self.val_epoch_freq != -1 and self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
            #     self.run_val()

            self.epoch += 1

        self.epoch -= 1

    def run_val(self, is_train=False):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader, is_train)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def average_dicts(self, dict_list):
        """
        对一组dict中相同的key求平均值

        Args:
            dict_list (list[dict]): 例如 [ {"loss1": 0.1, "loss2": 0.3}, {"loss1": 0.2, "loss2": 0.4} ]

        Returns:
            dict: key -> 平均值
        """
        sums = defaultdict(float)
        counts = defaultdict(int)

        for d in dict_list:
            for k, v in d.items():
                sums[k] += v
                counts[k] += 1

        return {k: sums[k] / counts[k] for k in sums}

    @torch.no_grad()
    def val_epoch(self, val_loader, is_train=False):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }

        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        metric_total = {}
        log_data_list = []
        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16

            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                        enabled=self.optim_conf.amp.enabled,
                        dtype=amp_type,
                ):
                    val_loss_dict, y_hat = self._step(
                        batch, self.model, phase, loss_meters, data_iter=data_iter
                    )
                    log_data_list.append(val_loss_dict)

            if is_train and "dtu" not in batch['seq_name'][0]:
                # fix the data iter num to view image
                if data_iter == 0:
                    image = self._get_val_tb_image(batch, y_hat)
                    self._tensorboard_write_image(image, phase, self.steps['train'])
                result_dict = cal_batch_results_by_md5_while_training(y_hat, batch, self.is_multiframe,
                                                                      use_ransac=self.use_ransac,
                                                                      scale_by_points=self.scale_by_points)
                metric_total.update(result_dict)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        # calculate eval metrics while training and write them to tensorboard
        if len(metric_total) > 0:
            metric_results = calculate_nested_average(metric_total)
            self._tensorboard_write_metrics(metric_results, phase, self.steps['train'])
            for key, value in metric_results.items():
                print(f"train steps val results:{self.steps[phase]} \n"
                      f"{key}: {value}")

        if len(log_data_list) > 0:
            avg_loss = self.average_dicts(log_data_list)
            avg_loss['extrinsics'] = batch['input_extrinsics']
            self._update_and_log_scalars(avg_loss, phase, self.steps['train'], loss_meters)
            self._log_tb_visuals(avg_loss, phase, self.steps['train'])

        return True

    def train_epoch(self, train_open_dataloader, train_own_dataloader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        learning_rate = AverageMeter("LR", self.device, ":.8f")
        data_times = []
        phase = 'train'

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }

        for config in self.gradient_clipper.configs:
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")

        progress = ProgressMeter(
            num_batches=len(train_open_dataloader) + len(train_own_dataloader),
            meters=[
                learning_rate,
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        iters_per_epoch = len(train_open_dataloader) + len(train_own_dataloader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )

        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        selected_idx_flag = 0

        if train_open_dataloader == []:
            train_loaders = [train_own_dataloader]
            selected_idx_flag = 1
        elif train_own_dataloader == []:
            train_loaders = [train_open_dataloader]
            selected_idx_flag = 1
        else:
            train_loaders = [train_open_dataloader, train_own_dataloader]

        iter_loaders = [iter(loader) for loader in train_loaders]

        if self.save_freq_while_training >= iters_per_epoch:
            self.save_freq_while_training = iters_per_epoch - 1

        for data_iter, _ in enumerate(range(iters_per_epoch)):
            if self.start_data_iter > data_iter:
                continue
            self.model.train()

            if selected_idx_flag != 1:
                selected_idx = random.randint(0, 1)
            else:
                selected_idx = 0
            selected_iter = iter_loaders[selected_idx]
            try:
                batch = next(selected_iter)
            except StopIteration:
                train_loaders.pop(selected_idx)
                iter_loaders.pop(selected_idx)
                if not train_loaders:
                    print("All DataLoaders exhausted, training finished.")
                    break

            # batch in enumerate(train_loader)
            if data_iter > limit_train_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # === LPIPS启用检查：在训练阶段且尚未启用时检查 ===
            if phase == "train" and not getattr(self, '_lpips_enabled', False):
                if self.epoch >= self.lpips_start_epoch:
                    if hasattr(self.loss, 'rgb_lpips_loss_fn'):
                        self.loss.rgb_lpips_loss_fn.set_perceptual_loss(True)
                        self._lpips_enabled = True
                        logging.info(f"LPIPS enabled at epoch {self.epoch + 1} (step {self.steps['train'] + 1})")
            # === LPIPS启用检查结束 ===

            accum_steps = self.accum_steps

            if accum_steps == 1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs

            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )

            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)
                    self.tb_writer.log(
                        os.path.join("Grad", f"Grad/{key}"),
                        grad_norm,
                        self.steps[phase],
                    )

            # Optimizer step
            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            learning_rate.update(optim.optimizer.param_groups[0]["lr"])
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            # evaluating while training in the epoch

            if self.save_freq_while_training != -1 and (self.steps[phase] + 1) % self.save_freq_while_training == 0:
                self.save_checkpoint(self.epoch, data_iter + 1,
                                     checkpoint_names=[
                                         f"ckpt_epoach_{self.epoch}_steps_{self.steps[phase] + 1}_dataiter_{data_iter}"])
            if self.val_freq_while_training != -1 and self.steps[phase] % self.val_freq_while_training == 0:
                self.run_val(is_train=True)

            if not torch.cuda.is_available():
                torch.npu.empty_cache()
            else:
                torch.cuda.empty_cache()

        return True

    def _run_steps_on_batch_chunks(
            self,
            chunked_batches: List[Any],
            phase: str,
            loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """

        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        # 可视化
        # b_input_images = chunked_batches[0]['input_images']

        # # 可视化 input_images 和 target_images
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import os
        
        # # 创建保存目录
        # save_dir = "/home/ma-user/work/h30079704/5_yinoneresearch/2_vggt_ascend/YinOneResearch/vis_vggt_attn/test_images"
        # os.makedirs(save_dir, exist_ok=True)

        # # 获取 input 图像
        # print(f"📊 input_images 形状: {b_input_images.shape}")
        # input_images = b_input_images[1]
        # T, V, H, W, C = input_images.shape

        # # 可视化 input_images - 一张图显示所有时间步和视角
        # fig, axes = plt.subplots(T, V, figsize=(4*V, 4*T))
            
        # # 处理子图维度
        # if T == 1 and V == 1:
        #     axes = np.array([[axes]])
        # elif T == 1:
        #     axes = axes.reshape(1, -1)
        # elif V == 1:
        #     axes = axes.reshape(-1, 1)
            
        # for t in range(T):
        #     for v in range(V):
        #         img_np = input_images[t, v]  # 直接使用numpy数组
                
        #         axes[t, v].imshow(img_np.cpu().permute(1, 2, 0).numpy())
        #         axes[t, v].set_title(f'T={t}, V={v}', fontsize=12)
        #         axes[t, v].axis('off')

        # plt.suptitle(f'Input Images - {T} Time Steps × {V} Views', fontsize=16)
        # plt.tight_layout()

        # # 保存图像
        # save_path = os.path.join(save_dir, 'input_visualization.png')
        # plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.close()

        # print(f"✅ 保存 input_visualization.png 到: {save_path}")
        # import pdb; pdb.set_trace()

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16

        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                        enabled=self.optim_conf.amp.enabled,
                        dtype=amp_type,
                ):
                    loss_dict, _ = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )

                loss = loss_dict["loss_total"]
                loss_key = f"Loss/{phase}_loss_total"
                batch_size = chunked_batch["input_images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)
                del loss

    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics",
            "cam_points", "world_points", "point_masks",
        ]
        string_keys = ["seq_name"]

        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor,
                                                torch.flip(original_tensor, dims=[1])],
                                               dim=0)

        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2

        return batch

    def _process_batch(self, batch: Mapping):

        if self.data_conf.get('train_open_dataset', None) is None:
            if self.data_conf.get('train_own_dataset', None) is not None:
                repeat_batch = self.data_conf.train_own_dataset.common_config.repeat_batch
            elif self.data_conf.get('train', None) is not None:
                repeat_batch = self.data_conf.train.common_config.repeat_batch
            else:
                repeat_batch = False
        else:
            repeat_batch = self.data_conf.train_open_dataset.common_config.repeat_batch

        if repeat_batch:
            batch = self._apply_batch_repetition(batch)

        # flag
        use_flow = batch["has_flow"][0].item()
        use_moge_depth = batch["has_moge_depth"][0].item()
        is_multiview_temporal = "input_extrinsics" in batch

        if is_multiview_temporal:
            # === Process INPUT data ===
            normalized_input_extrinsics, normalized_input_cam_points, normalized_input_world_points, normalized_input_depths, input_avg_scale = \
                normalize_camera_extrinsics_and_points_batch_for_multiview_and_temporal(
                    extrinsics=batch["input_extrinsics"],
                    cam_points=batch["input_cam_points"],
                    world_points=batch["input_world_points"],
                    depths=batch["input_depths"],
                    point_masks=batch["input_point_masks"],
                    scale_by_points=self.scale_by_points,
                )
            
            # Replace with normalized input data
            batch["input_extrinsics"] = normalized_input_extrinsics
            batch["input_cam_points"] = normalized_input_cam_points
            batch["input_world_points"] = normalized_input_world_points
            batch["input_depths"] = normalized_input_depths

            # === Process INPUT moge data ===
            if use_moge_depth:
                _, normalized_input_moge_cam_points, normalized_input_moge_world_points, normalized_input_moge_depths, input_avg_scale = \
                normalize_camera_extrinsics_and_points_batch_for_multiview_and_temporal(
                    extrinsics=batch["input_extrinsics"],
                    cam_points=batch["input_moge_cam_points"],
                    world_points=batch["input_moge_world_points"],
                    depths=batch["input_moge_depths"],
                    point_masks=batch["input_sky_masks"],
                    scale_by_points=self.scale_by_points,
                )

                batch["input_cam_points"] = normalized_input_moge_cam_points
                batch["input_world_points"] = normalized_input_moge_world_points
                batch["input_depths"] = normalized_input_moge_depths

            # Save original metric data for input
            # batch["metric_input_world_points"] = batch["input_world_points"]
            # batch["metric_input_depths"] = batch["input_depths"]
            # batch["metric_input_cam_points"] = batch["input_cam_points"]
            # batch["metric_input_extrinsics"] = batch["input_extrinsics"]
            # batch["metric_input_gt_points"] = utils3d.torch.depth_to_points(
            #     batch["input_depths"], intrinsics=batch["input_intrinsics"]
            # )
                
            
            # === Process TARGET data ===
            normalized_target_extrinsics, normalized_target_cam_points, normalized_target_world_points, normalized_target_depths, target_avg_scale = \
                normalize_camera_extrinsics_and_points_batch_for_multiview_and_temporal(
                    extrinsics=batch["target_extrinsics"],
                    cam_points=batch["target_cam_points"],
                    world_points=batch["target_world_points"],
                    depths=batch["target_depths"],
                    point_masks=batch["target_point_masks"],
                    scale_by_points=self.scale_by_points,
                )
                
            # Replace with normalized target data
            batch["target_extrinsics"] = normalized_target_extrinsics
            batch["target_cam_points"] = normalized_target_cam_points
            batch["target_world_points"] = normalized_target_world_points
            batch["target_depths"] = normalized_target_depths

            # === Process INPUT moge data ===
            if use_moge_depth:
                _, normalized_target_moge_cam_points, normalized_target_moge_world_points, normalized_target_moge_depths, target_avg_scale = \
                normalize_camera_extrinsics_and_points_batch_for_multiview_and_temporal(
                    extrinsics=batch["target_extrinsics"],
                    cam_points=batch["target_moge_cam_points"],
                    world_points=batch["target_moge_world_points"],
                    depths=batch["target_moge_depths"],
                    point_masks=batch["target_sky_masks"],
                    scale_by_points=False,
                    scale_by_avg_scale=target_avg_scale
                )
                
                batch["target_cam_points"] = normalized_target_moge_cam_points
                batch["target_world_points"] = normalized_target_moge_world_points
                batch["target_depths"] = normalized_target_moge_depths
            
            # import matplotlib.pyplot as plt
            # import torch
            # import numpy as np

            # # 提取数据 - [0,0,:] 表示 batch=0, time=0, 所有view
            # moge_views = normalized_target_moge_depths[0, 0, :]  # shape: [3, 196, 518]
            # target_views = normalized_target_depths[0, 0, :]     # shape: [3, 196, 518]

            # print(f"Moge views shape: {moge_views.shape}")  # [3, 196, 518]
            # print(f"Target views shape: {target_views.shape}")  # [3, 196, 518]

            # # 转换为numpy
            # moge_np = moge_views.detach().cpu().numpy()
            # target_np = target_views.detach().cpu().numpy()

            # # 为每个view创建对比图
            # for view_idx in range(moge_np.shape[0]):
            #     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                
            #     # 第一行：Moge深度图
            #     moge_view = moge_np[view_idx]  # [196, 518]
            #     im1 = axes[0, 0].imshow(moge_view, cmap='viridis')
            #     axes[0, 0].set_title(f'Moge Depth - View {view_idx}')
            #     axes[0, 0].axis('off')
            #     plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
                
            #     axes[0, 1].imshow(moge_view, cmap='gray')
            #     axes[0, 1].set_title(f'Moge Depth (Gray) - View {view_idx}')
            #     axes[0, 1].axis('off')
                
            #     # 第二行：Target深度图
            #     target_view = target_np[view_idx]  # [196, 518]
            #     im2 = axes[1, 0].imshow(target_view, cmap='viridis')
            #     axes[1, 0].set_title(f'Target Depth - View {view_idx}')
            #     axes[1, 0].axis('off')
            #     plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
                
            #     axes[1, 1].imshow(target_view, cmap='gray')
            #     axes[1, 1].set_title(f'Target Depth (Gray) - View {view_idx}')
            #     axes[1, 1].axis('off')
                
            #     # 差异图
            #     diff = moge_view - target_view
            #     im3 = axes[0, 2].imshow(diff, cmap='coolwarm', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            #     axes[0, 2].set_title(f'Difference - View {view_idx}')
            #     axes[0, 2].axis('off')
            #     plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
                
            #     # 绝对差异
            #     abs_diff = np.abs(diff)
            #     im4 = axes[1, 2].imshow(abs_diff, cmap='hot')
            #     axes[1, 2].set_title(f'Abs Diff - View {view_idx}')
            #     axes[1, 2].axis('off')
            #     plt.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)
                
            #     plt.tight_layout()
            #     plt.savefig(f'depth_comparison_view_{view_idx}.png', dpi=300, bbox_inches='tight')
            #     plt.show()
                
            #     # 打印统计信息
            #     print(f"\nView {view_idx} 统计:")
            #     print(f"  Moge范围: [{moge_view.min():.4f}, {moge_view.max():.4f}]")
            #     print(f"  Target范围: [{target_view.min():.4f}, {target_view.max():.4f}]")
            #     print(f"  MAE: {abs_diff.mean():.4f}, MaxAE: {abs_diff.max():.4f}")
            # import pdb; pdb.set_trace()
            # Save original metric data for target
            # batch["metric_target_world_points"] = batch["target_world_points"]
            # batch["metric_target_depths"] = batch["target_depths"]
            # batch["metric_target_cam_points"] = batch["target_cam_points"]
            # batch["metric_target_extrinsics"] = batch["target_extrinsics"]
            # batch["metric_target_gt_points"] = utils3d.torch.depth_to_points(
            #     batch["target_depths"], intrinsics=batch["target_intrinsics"]
            # )

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict, data_iter=None):
        """
        Performs a single forward pass, computes loss, and logs results.

        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        query_points = batch['tracks'][:, 0, :, :] if 'tracks' in batch else None
        y_hat = model(batch, query_points=query_points)

        # Loss computation
        loss_dict = self.loss(y_hat, batch)

        # Combine data for logging (训练阶段避免保留大张量)
        if phase == "train":
            # ✅ 只保留标量值和必要信息，不保留大张量
            log_data = {
                'loss_dict': loss_dict,
                'batch_meta': {k: v for k, v in batch.items() if not isinstance(v, torch.Tensor) or v.numel() < 1000},
                'input_extrinsics': batch['input_extrinsics']  # 用于 batch_size
            }
            self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
            self._log_tb_visuals(log_data, phase, self.steps[phase])

            # ✅ 训练结束后及时释放大张量引用
            if 'render_results' in y_hat:
                del y_hat['render_results']
            if 'depth' in y_hat:
                del y_hat['depth']
            if 'points' in y_hat:
                del y_hat['points']
            if 'world_points' in y_hat:
                del y_hat['world_points']
        else:
            # 验证阶段保留完整数据
            log_data = {**y_hat, **loss_dict, **batch}
            self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)

        self.steps[phase] += 1
        return loss_dict, y_hat

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)

        # ✅ 处理新的 log_data 结构
        if 'loss_dict' in data:
            loss_dict = data['loss_dict']
            batch_info = data['batch_meta']

            # 获取 batch_size
            batch_size = batch_info.get('input_extrinsics', torch.zeros(1)).shape[0] if 'input_extrinsics' in batch_info else 1

            for key in keys_to_log:
                if key in loss_dict:
                    value = loss_dict[key].item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]
                    loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                    if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                        self.tb_writer.log(f"Values/{phase}/{key}", value, step)
            return

        # 原有逻辑（验证阶段等）
        batch_size = data['input_extrinsics'].shape[0]

        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _tensorboard_write_image(self, image, phase: str, step: int):
        self.tb_writer.log_visuals(f"Metrics/{phase}/val_images", image[..., ::-1], step, dataformats='HWC')

    def _tensorboard_write_metrics(self, metric_dict: dict, phase: str, step: int):
        for key, value in metric_dict.items():
            self.tb_writer.log(f"Metrics/{phase}/{key}", value, step)

    def _get_val_tb_image(self, batch, preds):
        image_list = []
        h, w = batch['input_depths'][0, 0, ...].shape
        SCALE_BY_POINTS = os.getenv("SCALE_BY_POINTS", None)
        if SCALE_BY_POINTS is None:
            SCALE_BY_POINTS = False
        else:
            SCALE_BY_POINTS = True
        error_img = error_map(preds['depth'][0, 0, ..., 0], batch['input_depths'][0, 0, ...], h, w, SCALE_BY_POINTS)
        pred_vis = get_depth_vis(preds['depth'][0, 0, ..., 0], h, w, "pred")
        gt_vis = get_depth_vis(batch['input_depths'][0, 0, ...], h, w, "gt")
        ori_img = cv2.resize(
            (batch['input_images'].detach().cpu().numpy()[0, 0, ...].transpose(1, 2, 0)[..., ::-1] * 255).astype(int)
            , (h, w), interpolation=cv2.INTER_AREA)
        image_list.append(ori_img.astype(np.uint8))
        image_list.append(error_img)
        image_list.append(pred_vis)
        image_list.append(gt_vis)
        return create_2row_grid(image_list)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
                self.logging_conf.log_visuals
                and (phase in self.logging_conf.log_visual_frequency)
                and self.logging_conf.log_visual_frequency[phase] > 0
                and (step % self.logging_conf.log_visual_frequency[phase] == 0)
                and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                    len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
            isinstance(data, Sequence)
            and not isinstance(data, str)
            and len(data) > 0
            and isinstance(data[0], (str, int, float, bool))
    )


def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data


def error_map(pred, gt, h, w, SCALE_BY_POINTS, max_error=25.0):
    import torch.nn.functional as F
    gt = gt.detach().cpu().numpy()
    pred = cv2.resize(pred.detach().cpu().numpy(), (h, w), interpolation=cv2.INTER_AREA)
    if SCALE_BY_POINTS:
        pred = get_metric_depth_by_ransac(pred, gt)
    # jet = cv2.resize(cv2.imread('common/jet_color.png'), (120, 15))
    zero_mat = np.zeros((w, h, 3), np.uint8)
    error_map = 100.0 * np.abs(pred - gt) / (gt + 1e-7)
    mask_test = np.where((pred > 0.01) & (gt > 0.5) & (gt < 20) & (error_map < max_error))
    zero_mat[:, :, :][mask_test] = 1
    error_map = error_map * (250 / max_error)
    depth_image = error_map.astype(np.uint8)
    heat_img = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    heat_img1 = np.multiply(heat_img, zero_mat)
    # heat_img1 = cv2.resize(heat_img1, (768, 480))
    # heat_img1[10:25, 10:130, :] = jet
    # cv2.putText(heat_img1, "0%        {}%".format(int(max_error)), (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255))
    return heat_img1


def get_metric_depth_by_ransac(pred_depth, gt):
    mask_0 = np.logical_and(gt > 0, gt < 30)
    ransac = RANSACRegressor()
    # ransac.fit(pred_depth.reshape(-1, 1), gt.reshape(-1, ))
    ransac.fit(pred_depth[mask_0].reshape(-1, 1), gt[mask_0].reshape(-1, ))
    pred_depth = ransac.estimator_.coef_.item() * pred_depth + ransac.estimator_.intercept_.item()
    return pred_depth


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    from matplotlib.colors import ListedColormap
    # Construct the list colormap, with interpolated values for higher resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def get_depth_vis(depth, h, w, text="gt"):
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if text != "gt":
        depth = cv2.resize(depth, (h, w), interpolation=cv2.INTER_AREA)
    per_pred_disp = 1 / (depth + 1e-2)
    import matplotlib as mpl
    percent = 90
    per_mask = depth > 0
    vmax = np.percentile(per_pred_disp[per_mask], percent)
    if vmax > 0.9:
        vmax = np.percentile(
            per_pred_disp[per_mask], percent * 0.9)
    normalizer = mpl.colors.Normalize(
        vmin=per_pred_disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(
        norm=normalizer, cmap=high_res_colormap(cm.get_cmap('magma')))
    mask0 = np.repeat(per_mask[:, :, np.newaxis], 3, axis=2)
    colormap = (mapper.to_rgba(per_pred_disp)[:, :, :3] * 255 * mask0).astype(np.uint8)
    colormap_bgr = cv2.resize(colormap, (w, h))[:, :, ::-1]
    colormap_bgr = np.ascontiguousarray(colormap_bgr, dtype=np.uint8)
    cv2.putText(colormap_bgr, text, (w // 8, h // 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return colormap_bgr


def create_2row_grid(image_list):
    # 1. 分割图像列表
    mid_idx = len(image_list) // 2
    row1 = image_list[:mid_idx]  # 第一行图像
    row2 = image_list[mid_idx:]  # 第二行图像

    # 2. 水平拼接每行（无需调整尺寸）
    row1_combined = np.hstack(tuple(row1))  # 第一行水平拼接
    row2_combined = np.hstack(tuple(row2))  # 第二行水平拼接

    # 3. 垂直拼接两行
    grid = np.vstack((row1_combined, row2_combined))
    return grid
