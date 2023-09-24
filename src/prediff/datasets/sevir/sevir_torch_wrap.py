"""
Code is adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/e60ff41c7ad806277edc2a14a7a9f45585997bd7/src/earthformer/datasets/sevir/sevir_torch_wrap.py
Add data augmentation.
Only return "VIL" data in `torch.Tensor` format instead of `Dict`
"""
import os
from typing import Union, Dict, Sequence, Tuple, List
import numpy as np
import datetime
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from einops import rearrange
from lightning import LightningDataModule, seed_everything
from .sevir_dataloader import SEVIRDataLoader
from ...utils.path import default_dataset_sevir_dir, default_dataset_sevirlr_dir
from ..augmentation import TransformsFixRotation


def check_aws():
    r"""
    Check if aws cli is installed.
    """
    if os.system("which aws") != 0:
        raise RuntimeError("AWS CLI is not installed! Please install it first. See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")


def download_SEVIR(save_dir=None):
    r"""
    Downloaded dataset is saved in save_dir/sevir
    """

    check_aws()

    if save_dir is None:
        save_dir = default_dataset_sevir_dir
    else:
        save_dir = os.path.join(save_dir, "sevir")
    if os.path.exists(save_dir):
        raise FileExistsError(f"Path to save SEVIR dataset {save_dir} already exists!")
    else:
        os.makedirs(save_dir)
        os.system(f"aws s3 cp --no-sign-request s3://sevir/CATALOG.csv "
                  f"{os.path.join(save_dir, 'CATALOG.csv')}")
        os.system(f"aws s3 cp --no-sign-request --recursive s3://sevir/data/vil "
                  f"{os.path.join(save_dir, 'data', 'vil')}")


def download_SEVIRLR(save_dir=None):
    r"""
    Downloaded dataset is saved in save_dir/sevirlr
    """
    if save_dir is None:
        save_dir = default_dataset_sevirlr_dir
    else:
        save_dir = os.path.join(save_dir, "sevirlr")
    if os.path.exists(save_dir):
        raise FileExistsError(f"Path to save SEVIR-LR dataset {save_dir} already exists!")
    else:
        os.makedirs(save_dir)
        os.system(f"wget https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip "
                  f"-P {os.path.abspath(save_dir)}")
        os.system(f"unzip {os.path.join(save_dir, 'sevir_lr.zip')} "
                  f"-d {save_dir}")
        os.system(f"mv {os.path.join(save_dir, 'sevir_lr', '*')} "
                  f"{save_dir}\n"
                  f"rm -rf {os.path.join(save_dir, 'sevir_lr')}")


class SEVIRTorchDataset(TorchDataset):

    orig_dataloader_layout = "NHWT"
    orig_dataloader_squeeze_layout = orig_dataloader_layout.replace("N", "")
    aug_layout = "THW"

    def __init__(self,
                 seq_len: int = 25,
                 raw_seq_len: int = 49,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "THWC",
                 split_mode: str = "uneven",
                 sevir_catalog: Union[str, pd.DataFrame] = None,
                 sevir_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter = None,
                 catalog_filter = "default",
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout.replace("C", "1")
        self.ret_contiguous = ret_contiguous
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=["vil", ],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=1,
            layout=self.orig_dataloader_layout,
            num_shard=1,
            rank=0,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose)
        self.aug_mode = aug_mode
        if aug_mode == "0":
            self.aug = lambda x:x
        elif aug_mode == "1":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180),
            )
        elif aug_mode == "2":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270]),
            )
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        data = data_dict["vil"].squeeze(0)
        if self.aug_mode != "0":
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.aug_layout)}")
            data = self.aug(data)
            data = rearrange(data, f"{' '.join(self.aug_layout)} -> {' '.join(self.layout)}")
        else:
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.layout)}")
        if self.ret_contiguous:
            return data.contiguous()
        else:
            return data

    def __len__(self):
        return self.sevir_dataloader.__len__()


class SEVIRLightningDataModule(LightningDataModule):

    def __init__(self,
                 seq_len: int = 25,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "NTHWC",
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True,
                 # datamodule_only
                 dataset_name: str = "sevir",
                 sevir_dir: str = None,
                 start_date: Tuple[int] = None,
                 train_test_split_date: Tuple[int] = (2019, 6, 1),
                 end_date: Tuple[int] = None,
                 val_ratio: float = 0.1,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 seed: int = 0,
                 ):
        super(SEVIRLightningDataModule, self).__init__()
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        assert layout[0] == "N"
        self.layout = layout.replace("N", "")
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        if sevir_dir is not None:
            sevir_dir = os.path.abspath(sevir_dir)
        if dataset_name == "sevir":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevir_dir
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 384
            img_width = 384
        elif dataset_name == "sevirlr":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevirlr_dir
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'sevir' or 'sevirlr'.")
        self.dataset_name = dataset_name
        self.sevir_dir = sevir_dir
        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width
        # train val test split
        self.start_date = datetime.datetime(*start_date) \
            if start_date is not None else None
        self.train_test_split_date = datetime.datetime(*train_test_split_date) \
            if train_test_split_date is not None else None
        self.end_date = datetime.datetime(*end_date) \
            if end_date is not None else None
        self.val_ratio = val_ratio

    def prepare_data(self) -> None:
        if os.path.exists(self.sevir_dir):
            # Further check
            assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            if self.dataset_name == "sevir":
                download_SEVIR(save_dir=os.path.dirname(self.sevir_dir))
            elif self.dataset_name == "sevirlr":
                download_SEVIRLR(save_dir=os.path.dirname(self.sevir_dir))
            else:
                raise NotImplementedError

    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):
            sevir_train_val = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=True,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.start_date,
                end_date=self.train_test_split_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode=self.aug_mode,
                ret_contiguous=self.ret_contiguous,)
            self.sevir_train, self.sevir_val = random_split(
                dataset=sevir_train_val,
                lengths=[1 - self.val_ratio, self.val_ratio],
                generator=torch.Generator().manual_seed(self.seed))
        if stage in (None, "test"):
            self.sevir_test = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=False,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.train_test_split_date,
                end_date=self.end_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode="0",
                ret_contiguous=self.ret_contiguous,)

    def train_dataloader(self):
        return DataLoader(self.sevir_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.sevir_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.sevir_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)
