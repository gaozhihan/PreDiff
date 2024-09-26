import warnings
import os
from typing import Union, Dict, Sequence, Tuple, List
import numpy as np
import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning import LightningDataModule, seed_everything

from ...utils.path import default_pd_path, default_dataset_hko_dir
from ..utils import random_split
from .hko_torch import HKODataset


class HKOLightningDataModule(LightningDataModule):

    def __init__(self,
                 pd_path=None,
                 train_test_split_datetime="2015-01-01 00:00:00",
                 val_ratio=0.1,
                 train_pd_path=None,
                 val_pd_path=None,
                 test_pd_path=None,
                 seq_len=30,
                 max_consecutive_missing=2,
                 stride=None,
                 height=None,
                 width=None,
                 base_freq='6min',
                 downscaling_scale: Sequence[int] = None,
                 interpolate_resize: Sequence[int] = None,
                 norm_mode="01",
                 aug_mode="0",
                 # LightningDataModule
                 batch_size: int = 1,
                 num_workers: int = 1,
                 seed: int = 0,
                 weighted_sampler: str = "0",
                 ):
        """
        Parameters
        ----------
        pd_path:  str
            path of the saved pandas dataframe of all (train and test) data.
        train_test_split_datetime:  str
            The datetime string of the split time between train and test data.
        val_ratio:  float
        seq_len:    int
        max_consecutive_missing:    int
            The maximum consecutive missing frames
        stride: int or None, optional
        height: int or None, optional
            Spatial dimension of the raw data frames, not the one after downscaling
        width:  int or None, optional
            Spatial dimension of the raw data frames, not the one after downscaling
        base_freq : str, optional
        downscaling_scale:  Sequence[int]
            [s_height, s_width] are factors for downscaling along height and width dimensions.
        interpolate_resize: Sequence[int]
            [rs_height, rs_width] are the target spatial dimensions
            [optionally after downscaling, and then] after interpolation.
        norm_mode:  str
            The values of the original data are from 0 to 255.
            "01" by default. Rescale the value range to [0, 1].
        aug_mode:   str
            "0" by default, which uses no augmentation
        weighted_sampler:   str
            weighted sampling for training data
            "0" by default, which uses no weighted sampler
        """
        super().__init__()
        if pd_path is None:
            pd_path = default_pd_path
        else:
            pd_path = os.path.join(default_dataset_hko_dir, pd_path)
        self.pd_path = pd_path
        if train_pd_path is None and val_pd_path is None and test_pd_path is None:
            self._df = pd.read_pickle(pd_path)
            self.train_test_split_ind = (self._df.index < pd.to_datetime(train_test_split_datetime)).sum()
            self.val_ratio = val_ratio
            self.auto_split_flag = True
        else:
            assert train_pd_path is not None and val_pd_path is not None and test_pd_path is not None
            self.train_pd_path = os.path.join(default_dataset_hko_dir, train_pd_path)
            self.val_pd_path = os.path.join(default_dataset_hko_dir, val_pd_path)
            self.test_pd_path = os.path.join(default_dataset_hko_dir, test_pd_path)
            self.train_test_split_ind = None
            self.val_ratio = None
            self.auto_split_flag = False
        self.seq_len = seq_len
        self.max_consecutive_missing = max_consecutive_missing
        self.stride = stride
        self.height = height
        self.width = width
        self.base_freq = base_freq
        self.downscaling_scale = downscaling_scale
        self.interpolate_resize = interpolate_resize
        self.norm_mode = norm_mode
        self.aug_mode = aug_mode
        # LightningDataModule
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.weighted_sampler = weighted_sampler

    def prepare_data(self) -> None:
        warnings.warn(f"The automatic downloading of HKO dataset is not supported!")

    def setup(self, stage: str = None) -> None:
        seed_everything(seed=self.seed)
        if self.auto_split_flag:
            if stage in (None, "fit"):
                self.hko_train_val = HKODataset(
                    pd_path=self.pd_path,
                    sample_mode="random",
                    seq_len=self.seq_len,
                    max_consecutive_missing=self.max_consecutive_missing,
                    begin_ind=None,
                    end_ind=self.train_test_split_ind,
                    stride=None,
                    height=self.height,
                    width=self.width,
                    base_freq=self.base_freq,
                    downscaling_scale=self.downscaling_scale,
                    interpolate_resize=self.interpolate_resize,
                    norm_mode=self.norm_mode,
                    aug_mode=self.aug_mode
                )
                [self.hko_train, self.hko_val], [self.train_indices, _] = random_split(
                    dataset=self.hko_train_val,
                    lengths=[1 - self.val_ratio, self.val_ratio],
                    generator=torch.Generator().manual_seed(self.seed),
                    return_indices=True, )
            if stage in (None, "test"):
                self.hko_test = HKODataset(
                    pd_path=self.pd_path,
                    sample_mode="sequent",
                    seq_len=self.seq_len,
                    max_consecutive_missing=self.max_consecutive_missing,
                    begin_ind=self.train_test_split_ind,
                    end_ind=None,
                    stride=self.stride,
                    height=self.height,
                    width=self.width,
                    base_freq=self.base_freq,
                    downscaling_scale=self.downscaling_scale,
                    interpolate_resize=self.interpolate_resize,
                    norm_mode=self.norm_mode,
                )
        else:
            if stage in (None, "fit"):
                self.hko_train = HKODataset(
                    pd_path=self.train_pd_path,
                    sample_mode="random",
                    seq_len=self.seq_len,
                    max_consecutive_missing=self.max_consecutive_missing,
                    begin_ind=None,
                    end_ind=None,
                    stride=None,
                    height=self.height,
                    width=self.width,
                    base_freq=self.base_freq,
                    downscaling_scale=self.downscaling_scale,
                    interpolate_resize=self.interpolate_resize,
                    norm_mode=self.norm_mode,
                    aug_mode=self.aug_mode
                )
                self.hko_val = HKODataset(
                    pd_path=self.val_pd_path,
                    sample_mode="sequent",
                    seq_len=self.seq_len,
                    max_consecutive_missing=self.max_consecutive_missing,
                    begin_ind=None,
                    end_ind=None,
                    stride=self.stride,
                    height=self.height,
                    width=self.width,
                    base_freq=self.base_freq,
                    downscaling_scale=self.downscaling_scale,
                    interpolate_resize=self.interpolate_resize,
                    norm_mode=self.norm_mode,
                    aug_mode=self.aug_mode
                )
            if stage in (None, "test"):
                self.hko_test = HKODataset(
                    pd_path=self.test_pd_path,
                    sample_mode="sequent",
                    seq_len=self.seq_len,
                    max_consecutive_missing=self.max_consecutive_missing,
                    begin_ind=None,
                    end_ind=None,
                    stride=self.stride,
                    height=self.height,
                    width=self.width,
                    base_freq=self.base_freq,
                    downscaling_scale=self.downscaling_scale,
                    interpolate_resize=self.interpolate_resize,
                    norm_mode=self.norm_mode,
                )

    def train_dataloader(self):
        if self.weighted_sampler == "0":
            return DataLoader(self.hko_train,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers)
        elif self.weighted_sampler == "1":
            weights = [self.hko_train_val.all_avg_int_clips[ele] for ele in self.train_indices]  # list indices must be integers or slices, not list
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=self.num_train_samples,
                replacement=True)
            return DataLoader(self.hko_train,
                              batch_size=self.batch_size,
                              sampler=sampler,
                              num_workers=self.num_workers)
        elif float(self.weighted_sampler) > 1.0:
            weights_threshold = float(self.weighted_sampler)
            weights = [self.hko_train_val.all_avg_int_clips[ele] for ele in self.train_indices]
            weights = np.array(weights)
            weights[weights > weights_threshold] = weights_threshold  # for single frame
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=self.num_train_samples,
                replacement=True)
            return DataLoader(self.hko_train,
                              batch_size=self.batch_size,
                              sampler=sampler,
                              num_workers=self.num_workers)
        else:
            raise NotImplementedError("Weighted sampler is not implemented!")

    def val_dataloader(self):
        return DataLoader(self.hko_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.hko_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.hko_train)

    @property
    def num_val_samples(self):
        return len(self.hko_val)

    @property
    def num_test_samples(self):
        return len(self.hko_test)
        