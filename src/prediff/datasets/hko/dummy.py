import warnings
import os
from typing import Union, Dict, Sequence, Tuple, List
import numpy as np
from skimage.measure import block_reduce
import datetime
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange
from torchvision import transforms
from lightning import LightningDataModule, seed_everything

from ..augmentation import TransformsFixRotation


class DummyHKODataset(Dataset):
    """
    Dummy HKO-7 PyTorch Dataset for fast debugging
    """
    default_height = 480
    default_width = 480

    def __init__(self,
                 pd_path=None,
                 exclude_mask_path=None,
                 png_file_dir=None,
                 mask_file_dir=None,
                 sample_mode="random",
                 seq_len=30,
                 max_consecutive_missing=2,
                 begin_ind=None,
                 end_ind=None,
                 stride=None,
                 height=None,
                 width=None,
                 base_freq='6min',
                 downscaling_scale: Sequence[int] = None,
                 interpolate_resize: Sequence[int] = None,
                 norm_mode="01",
                 aug_mode="0",
                 ):
        """Random sample: sample a random clip that will not violate the max_missing frame_num criteria
        Sequent sample: sample a clip from the beginning of the time.
                        Everytime, the clips from {T_begin, T_begin + 6min, ..., T_begin + (seq_len-1) * 6min} will be used
                        The begin datetime will move forward by adding stride: T_begin += 6min * stride
                        Once the clips violates the maximum missing number criteria, the starting
                         point will be moved to the next datetime that does not violate the missing_frame criteria

        Parameters
        ----------
        pd_path:    str
            path of the saved pandas dataframe
        sample_mode:    str
            Can be "random" or "sequent"
        seq_len:    int
        max_consecutive_missing:    int
            The maximum consecutive missing frames
        begin_ind:  int
            Index of the begin frame
        end_ind:    int
            Index of the end frame
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
        """
        if width is None:
            width = self.default_width
        if height is None:
            height = self.default_height
        self._seq_len = seq_len
        self._width = width
        self._height = height
        self._stride = stride

        self.downscaling_scale = downscaling_scale
        self.interpolate_resize = interpolate_resize
        self.norm_mode = norm_mode
        self.aug_mode = aug_mode
        if aug_mode == "0":
            self.aug = lambda x: x
        elif aug_mode == "1":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180, fill=0.),
            )
        elif aug_mode == "2":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270], fill=0.),
            )
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item:   int

        Returns
        -------
        frame_dat:  torch.Tensor
            Shape: (seq_len, height, width)
        mask_dat:   torch.Tensor
            Shape: (seq_len, height, width)
        """
        frame_dat = np.random.rand(self._seq_len, self._height, self._width).astype(np.float32)
        mask_dat = np.random.rand(self._seq_len, self._height, self._width).astype(np.float32)
        if self.downscaling_scale is not None:
            # TODO: more specific downscaling strategy
            frame_dat = block_reduce(frame_dat*mask_dat,
                                     block_size=(1, *self.downscaling_scale),
                                     func=np.max)
            mask_dat = block_reduce(mask_dat,
                                    block_size=(1, *self.downscaling_scale),
                                    func=np.min)
        frame_dat = torch.from_numpy(frame_dat)
        mask_dat = torch.from_numpy(mask_dat)
        if self.interpolate_resize is not None:
            frame_dat = rearrange(frame_dat, "t h w -> t 1 h w")
            mask_dat = rearrange(mask_dat, "t h w -> t 1 h w")
            frame_dat = F.interpolate(input=frame_dat,
                                      size=self.interpolate_resize,
                                      mode="nearest", )
            mask_dat = F.interpolate(input=mask_dat,
                                     size=self.interpolate_resize,
                                     mode="nearest", )
            frame_dat = rearrange(frame_dat, "t 1 h w -> t h w")
            mask_dat = rearrange(mask_dat, "t 1 h w -> t h w")
        if self.aug_mode != "0":
            aug_data = self.aug(torch.cat([frame_dat, mask_dat], dim=0))
            frame_dat, mask_dat = torch.split(aug_data,
                                              split_size_or_sections=[self._seq_len, self._seq_len],
                                              dim=0)
        return frame_dat, mask_dat

    def __len__(self):
        return 1000


class DummyHKOLightningDataModule(LightningDataModule):

    def __init__(self,
                 pd_path=None,
                 train_test_split_datetime="2015-01-01 00:00:00",
                 train_pd_path=None,
                 test_pd_path=None,
                 val_ratio=0.1,
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
                 **kwargs,
                 ):
        """
        Parameters
        ----------
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
        """
        super().__init__()
        self.val_ratio = val_ratio
        self.seq_len = seq_len
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

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):
            hko_train_val = DummyHKODataset(
                sample_mode="random",
                seq_len=self.seq_len,
                stride=None,
                height=self.height,
                width=self.width,
                base_freq=self.base_freq,
                downscaling_scale=self.downscaling_scale,
                interpolate_resize=self.interpolate_resize,
                norm_mode=self.norm_mode,
                aug_mode=self.aug_mode
            )
            self.hko_train, self.hko_val = random_split(
                dataset=hko_train_val,
                lengths=[1 - self.val_ratio, self.val_ratio],
                generator=torch.Generator().manual_seed(self.seed))
        if stage in (None, "test"):
            self.hko_test = DummyHKODataset(
                sample_mode="sequent",
                seq_len=self.seq_len,
                stride=self.stride,
                height=self.height,
                width=self.width,
                base_freq=self.base_freq,
                downscaling_scale=self.downscaling_scale,
                interpolate_resize=self.interpolate_resize,
                norm_mode=self.norm_mode,
            )

    def train_dataloader(self):
        return DataLoader(self.hko_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

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
