"""
Code is adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/e60ff41c7ad806277edc2a14a7a9f45585997bd7/src/earthformer/datasets/sevir/sevir_torch_wrap.py
Add data augmentation.
Only return "VIL" data in `torch.Tensor` format instead of `Dict`
"""
import os
import random
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
from .sevir_dataloader import SEVIRDataLoader, NPYSEVIRDataLoader
from ...utils.path import default_dataset_sevir_dir, default_dataset_sevirlr_dir
from ..augmentation import TransformsFixRotation

import glob

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
                 ret_contiguous: bool = True,
                 sevir_dataloader: SEVIRDataLoader = None):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout.replace("C", "1")
        self.ret_contiguous = ret_contiguous
        self.sevir_dataloader = sevir_dataloader

        # np_dir = "/home/user01/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/data_npy"
        # self.sevir_dataloader = sevir_dataloader if sevir_dataloader is not None else NPYSEVIRDataLoader(npy_dir=np_dir, seq_len=25, raw_seq_len=25, stride=1)
        
        # self.sevir_dataloader = SEVIRDataLoader(
        #     data_types=["vil", ],
        #     seq_len=seq_len,
        #     raw_seq_len=raw_seq_len,
        #     sample_mode=sample_mode,
        #     stride=stride,
        #     batch_size=1,
        #     layout=self.orig_dataloader_layout,
        #     num_shard=1,
        #     rank=0,
        #     split_mode=split_mode,
        #     sevir_catalog=sevir_catalog,
        #     sevir_data_dir=sevir_data_dir,
        #     start_date=start_date,
        #     end_date=end_date,
        #     datetime_filter=datetime_filter,
        #     catalog_filter=catalog_filter,
        #     shuffle=shuffle,
        #     shuffle_seed=shuffle_seed,
        #     output_type=output_type,
        #     preprocess=preprocess,
        #     rescale_method=rescale_method,
        #     downsample_dict=None,
        #     verbose=verbose)
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
                 # 新增参数
                 npy_dir: str = None,
                 npy_num_samples: int = None,
                 random_split: bool = False,
                 split_seed: int = 0,
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
        # 保存新增参数为实例属性
        self.npy_dir = npy_dir or sevir_dir
        self.npy_num_samples = npy_num_samples
        self.random_split = random_split
        self.split_seed = split_seed
        
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
            #===================1109先粗暴改了这里========================
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        elif dataset_name == "sevirlr_npy13":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevirlr_dir
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 13
            interval_real_time = 10
            img_height = 128
            img_width = 128
        elif dataset_name == "sevirlr_npy30":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevirlr_dir
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 30
            interval_real_time = 5
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
            print("SEVIR data directory found. Skip downloading.")
            # assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            # assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            if self.dataset_name == "sevir":
                download_SEVIR(save_dir=os.path.dirname(self.sevir_dir))
            elif self.dataset_name == "sevirlr":
                download_SEVIRLR(save_dir=os.path.dirname(self.sevir_dir))
            else:
                raise NotImplementedError

    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        
        # 如果目录中包含 .npy，则采用 NPY workflow 并按 7:1.5:1.5 划分
        # 查找 npy 文件（优先使用 self.npy_dir）
        npy_dir = getattr(self, "npy_dir", None) or self.sevir_dir if hasattr(self, "sevir_dir") else None
        if npy_dir and os.path.isdir(npy_dir):
            npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        # 如果指定了要使用的样本数，截取前 n（或全部）
        if self.npy_num_samples:
            npy_files = npy_files[: int(self.npy_num_samples)]
        # 可选：随机打乱后再 split
        if self.random_split:
            # import random
            rnd = random.Random(self.split_seed)
            rnd.shuffle(npy_files)

        # npy_files = []
        print('**********')
        
        #=======================251016GPT改======================================
        if len(npy_files) > 0:
            print('-----------')
            n_files = len(npy_files)
            raw_seq_len = self.raw_seq_len
            seq_len = self.seq_len
            stride = self.stride
            # 每个文件可产生的滑窗序列数量
            num_seq_per_file = 1 + (raw_seq_len - seq_len) // stride
            total_possible_samples = n_files * num_seq_per_file
            print(f"[DEBUG] len(npy_files)={len(npy_files)}, raw_seq_len={self.raw_seq_len}, seq_len={self.seq_len}, stride={self.stride}")


            # ✅ (1) 从配置文件直接选定多少个样本，而不是多少个文件
            npy_num_samples = getattr(self, "npy_num_samples", None)
            if npy_num_samples is not None and npy_num_samples < total_possible_samples:
                max_samples = int(npy_num_samples)
            else:
                max_samples = total_possible_samples

            print(f"[INFO] total possible samples: {total_possible_samples}, use: {max_samples}")

            # 构造完整的 (filename, seq_idx) 列表
            all_samples = []
            for fidx, fname in enumerate(npy_files):
                for seq_idx in range(num_seq_per_file):
                    all_samples.append((fname, seq_idx))
                    if len(all_samples) >= max_samples:
                        break
                if len(all_samples) >= max_samples:
                    break

            # # ✅ (2) 按配置文件比例划分 train/val/test
            # train_ratio = 0.8
            # val_ratio = 0.1
            # test_ratio = 0.1
            # if hasattr(self, "val_ratio") and self.val_ratio is not None:
            #     val_ratio = float(self.val_ratio)
            #     # 自动保持三者和为1
            #     train_ratio = 1.0 - 2 * val_ratio
            #     test_ratio = val_ratio

            # n_total = len(all_samples)
            
            
            # # ✅ 特殊情况处理：样本过少时直接复用
            # if n_total <= 1:
            #     print(f"[WARN] Only {n_total} sample(s) found — using same file for train/val/test.")
            #     train_samples = val_samples = test_samples = all_samples
            # else:
            #     n_train = int(n_total * train_ratio)
            #     n_val = int(n_total * val_ratio)
            #     n_test = n_total - n_train - n_val

            #     # ✅ 确保至少各一条（防止空 split）
            #     if n_train == 0 and n_total > 0: n_train = 1
            #     if n_val == 0 and n_total > 1: n_val = 1
            #     if n_test == 0 and n_total > 2: n_test = 1

            #     train_samples = all_samples[:n_train]
            #     val_samples = all_samples[n_train:n_train + n_val]
            #     test_samples = all_samples[n_train + n_val:]

            # print(f"[Split] train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
            
            # === 2. 先固定 test，再对 train+val 随机划分 ===
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1
            if hasattr(self, "val_ratio") and self.val_ratio is not None:
                val_ratio = float(self.val_ratio)
                train_ratio = 1.0 - 2 * val_ratio
                test_ratio = val_ratio

            n_total = len(all_samples)

            if n_total <= 1:
                print(f"[WARN] Only {n_total} sample(s) found — using same file for train/val/test.")
                train_samples = val_samples = test_samples = all_samples
            else:
                # 先按比例算出 test 数量（固定最后一段为 test）
                n_test = int(n_total * test_ratio)
                if n_test == 0:
                    n_test = 1  # 至少保留一个样本做 test
                    
                n_train = int(n_total * train_ratio)
                n_val = int(n_total * val_ratio)
                n_test = n_total - n_train - n_val

                n_trainval = n_train + n_val
                if n_trainval <= 1:
                    # 样本太少，干脆全部复用
                    print(f"[WARN] Only {n_total} sample(s) for train+val — using same for all splits.")
                    train_samples = val_samples = test_samples = all_samples
                else:
                    # 按时间顺序：前面是 train+val，最后 n_test 是 test（固定不变）
                    trainval_samples = all_samples[:n_trainval]
                    test_samples = all_samples[n_trainval:]

                    # 在 train+val 内部用 split_seed 打乱，保证可复现
                    rnd = random.Random(self.split_seed)
                    rnd.shuffle(trainval_samples)

                    # train:val 再按原比例拆（在 trainval 内部）
                    # 注意这里只用 train_ratio 和 val_ratio 的相对占比
                    tv_sum = train_ratio + val_ratio
                    if tv_sum <= 0:
                        # 极端情况，默认 9:1
                        train_ratio_eff = 0.9
                        val_ratio_eff = 0.1
                    else:
                        train_ratio_eff = train_ratio / tv_sum
                        val_ratio_eff = val_ratio / tv_sum

                    n_train = int(n_trainval * train_ratio_eff)
                    n_val = n_trainval - n_train

                    if n_train == 0 and n_trainval > 0:
                        n_train = 1
                        n_val = n_trainval - 1
                    if n_val == 0 and n_trainval > 1:
                        n_val = 1
                        n_train = n_trainval - 1

                    train_samples = trainval_samples[:n_train]
                    val_samples = trainval_samples[n_train:]

            print(f"[Split] train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")


            # ✅ (3) 导出CSV记录
            split_records = []
            for split_name, subset in zip(["train", "val", "test"], [train_samples, val_samples, test_samples]):
                for idx, (fname, seq_idx) in enumerate(subset):
                    split_records.append({
                        "split": split_name,
                        "global_idx": idx,
                        "filename": os.path.basename(fname),
                        "seq_idx": seq_idx,
                        "full_path": fname
                    })
            record_path = os.path.join(self.sevir_dir, "split_record.csv")
            pd.DataFrame(split_records).to_csv(record_path, index=False)
            print(f"[INFO] Split record saved to {record_path}")

            # 构建三个 NPYSEVIRDataLoader
            def build_loader(subset_samples):
                # 按文件分组，保证 NPYLoader 接口不变
                from collections import defaultdict
                file_groups = defaultdict(list)
                for fname, seq_idx in subset_samples:
                    file_groups[fname].append(seq_idx)

                # 创建 pseudo-filelist
                file_list = list(file_groups.keys())
                loader = NPYSEVIRDataLoader(
                    file_list=file_list,
                    seq_len=self.seq_len,
                    raw_seq_len=self.raw_seq_len,
                    stride=self.stride,
                    batch_size=1,
                    layout='NHWT',
                    output_type=self.output_type,
                    preprocess=self.preprocess,
                    rescale_method=self.rescale_method
                )
                # 额外保存 file_groups 信息到 loader（方便 debug）
                loader.sample_indices = file_groups
                return loader

            train_loader = build_loader(train_samples)
            val_loader = build_loader(val_samples)
            test_loader = build_loader(test_samples)

            # 用 SEVIRTorchDataset 封装
            self.sevir_train = SEVIRTorchDataset(
                seq_len=self.seq_len,
                raw_seq_len=self.raw_seq_len,
                sample_mode='sequent',
                stride=self.stride,
                layout=self.layout,
                split_mode='uneven',
                sevir_dataloader=train_loader,
                shuffle=True,
                ret_contiguous=self.ret_contiguous,
                aug_mode=self.aug_mode
            )
            self.sevir_val = SEVIRTorchDataset(
                seq_len=self.seq_len,
                raw_seq_len=self.raw_seq_len,
                sample_mode='sequent',
                stride=self.stride,
                layout=self.layout,
                split_mode='uneven',
                sevir_dataloader=val_loader,
                shuffle=False,
                ret_contiguous=self.ret_contiguous,
                aug_mode='0'
            )
            self.sevir_test = SEVIRTorchDataset(
                seq_len=self.seq_len,
                raw_seq_len=self.raw_seq_len,
                sample_mode='sequent',
                stride=self.stride,
                layout=self.layout,
                split_mode='uneven',
                sevir_dataloader=test_loader,
                shuffle=False,
                ret_contiguous=self.ret_contiguous,
                aug_mode='0'
            )

        
        #=========================251015GPT改 可自定义npy文件数目 成功运行======================================
        # if os.path.isdir(self.sevir_dir):
        #     npy_files = sorted(glob.glob(os.path.join(self.sevir_dir, "*.npy")))
        # if len(npy_files) > 0:
        #     print('-----------')
        #     n = len(npy_files)
        #     n_train = int(n * 0.70)
        #     n_val = int(n * 0.15)
        #     n_test = n - n_train - n_val
        #     train_files = npy_files[:n_train]
        #     val_files = npy_files[n_train:n_train + n_val]
        #     test_files = npy_files[n_train + n_val:]
        #     # 创建对应的 NPY loaders
        #     train_loader = NPYSEVIRDataLoader(file_list=train_files,
        #                                       seq_len=self.seq_len,
        #                                       raw_seq_len=self.raw_seq_len,
        #                                       stride=self.stride,
        #                                       batch_size=1,
        #                                       layout='NHWT',
        #                                       output_type=self.output_type,
        #                                       preprocess=self.preprocess,
        #                                       rescale_method=self.rescale_method)
        #     val_loader = NPYSEVIRDataLoader(file_list=val_files,
        #                                     seq_len=self.seq_len,
        #                                     raw_seq_len=self.raw_seq_len,
        #                                     stride=self.stride,
        #                                     batch_size=1,
        #                                     layout='NHWT',
        #                                     output_type=self.output_type,
        #                                     preprocess=self.preprocess,
        #                                     rescale_method=self.rescale_method)
        #     test_loader = NPYSEVIRDataLoader(file_list=test_files,
        #                                      seq_len=self.seq_len,
        #                                      raw_seq_len=self.raw_seq_len,
        #                                      stride=self.stride,
        #                                      batch_size=1,
        #                                      layout='NHWT',
        #                                      output_type=self.output_type,
        #                                      preprocess=self.preprocess,
        #                                      rescale_method=self.rescale_method)
        #     # 用 SEVIRTorchDataset 包装（利用传入的 sevir_dataloader）
        #     self.sevir_train = SEVIRTorchDataset(seq_len=self.seq_len,
        #                                          raw_seq_len=self.raw_seq_len,
        #                                          sample_mode='sequent',
        #                                          stride=self.stride,
        #                                          layout=self.layout,
        #                                          split_mode='uneven',
        #                                          sevir_dataloader=train_loader,
        #                                          shuffle=True,
        #                                          ret_contiguous=self.ret_contiguous,
        #                                          aug_mode=self.aug_mode)
        #     self.sevir_val = SEVIRTorchDataset(seq_len=self.seq_len,
        #                                        raw_seq_len=self.raw_seq_len,
        #                                        sample_mode='sequent',
        #                                        stride=self.stride,
        #                                        layout=self.layout,
        #                                        split_mode='uneven',
        #                                        sevir_dataloader=val_loader,
        #                                        shuffle=False,
        #                                        ret_contiguous=self.ret_contiguous,
        #                                        aug_mode='0')
        #     self.sevir_test = SEVIRTorchDataset(seq_len=self.seq_len,
        #                                         raw_seq_len=self.raw_seq_len,
        #                                         sample_mode='sequent',
        #                                         stride=self.stride,
        #                                         layout=self.layout,
        #                                         split_mode='uneven',
        #                                         sevir_dataloader=test_loader,
        #                                         shuffle=False,
        #                                         ret_contiguous=self.ret_contiguous,
        #                                         aug_mode='0')

        
        # if stage in (None, "fit"):
        #     sevir_train_val = SEVIRTorchDataset(
        #         sevir_catalog=self.catalog_path,
        #         sevir_data_dir=self.raw_data_dir,
        #         raw_seq_len=self.raw_seq_len,
        #         split_mode="uneven",
        #         shuffle=True,
        #         seq_len=self.seq_len,
        #         stride=self.stride,
        #         sample_mode=self.sample_mode,
        #         layout=self.layout,
        #         start_date=self.start_date,
        #         end_date=self.train_test_split_date,
        #         output_type=self.output_type,
        #         preprocess=self.preprocess,
        #         rescale_method=self.rescale_method,
        #         verbose=self.verbose,
        #         aug_mode=self.aug_mode,
        #         ret_contiguous=self.ret_contiguous,)
        #     self.sevir_train, self.sevir_val = random_split(
        #         dataset=sevir_train_val,
        #         lengths=[1 - self.val_ratio, self.val_ratio],
        #         generator=torch.Generator().manual_seed(self.seed))
        # if stage in (None, "test"):
        #     self.sevir_test = SEVIRTorchDataset(
        #         sevir_catalog=self.catalog_path,
        #         sevir_data_dir=self.raw_data_dir,
        #         raw_seq_len=self.raw_seq_len,
        #         split_mode="uneven",
        #         shuffle=False,
        #         seq_len=self.seq_len,
        #         stride=self.stride,
        #         sample_mode=self.sample_mode,
        #         layout=self.layout,
        #         start_date=self.train_test_split_date,
        #         end_date=self.end_date,
        #         output_type=self.output_type,
        #         preprocess=self.preprocess,
        #         rescale_method=self.rescale_method,
        #         verbose=self.verbose,
        #         aug_mode="0",
        #         ret_contiguous=self.ret_contiguous,)

    def train_dataloader(self):
        return DataLoader(self.sevir_train,
                          batch_size=self.batch_size,
                          shuffle=False,
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

    # @property
    # def num_train_samples(self):
    #     return len(self.sevir_train)

    # @property
    # def num_val_samples(self):
    #     return len(self.sevir_val)

    # @property
    # def num_test_samples(self):
    #     return len(self.sevir_test)
    
    @property
    def num_train_samples(self):
        if hasattr(self, "sevir_train"):
            return len(self.sevir_train)
        return 0

    @property
    def num_val_samples(self):
        if hasattr(self, "sevir_val"):
            return len(self.sevir_val)
        return 0

    @property
    def num_test_samples(self):
        if hasattr(self, "sevir_test"):
            return len(self.sevir_test)
        return 0
