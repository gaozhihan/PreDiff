from typing import Sequence
import os
import pandas as pd
import numpy as np
from skimage.measure import block_reduce
import bisect
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from einops import rearrange
from torchvision import transforms


from ...utils.path import (
    default_dataset_hko_dir,
    default_pd_path,
    default_exclude_mask_path,
    default_png_file_dir,
    default_mask_file_dir,
)
from .image import quick_read_frames
from .mask import quick_read_masks
from ..augmentation import TransformsFixRotation


class HKODataset(Dataset):
    """
    The HKO-7 PyTorch Dataset
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
        if pd_path is None:
            pd_path = default_pd_path
        if exclude_mask_path is None:
            exclude_mask_path = default_exclude_mask_path
        if png_file_dir is None:
            png_file_dir = default_png_file_dir
        if mask_file_dir is None:
            mask_file_dir = default_mask_file_dir
        self._df = pd.read_pickle(pd_path)
        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._df_index_set = frozenset([self._df.index[i] for i in range(len(self._df))])
        self._exclude_mask = self.get_exclude_mask(exclude_mask_path)
        self.png_file_dir = png_file_dir
        self.mask_file_dir = mask_file_dir
        self._seq_len = seq_len
        self._width = width
        self._height = height
        self._stride = stride
        self._max_consecutive_missing = max_consecutive_missing
        self._base_freq = base_freq
        self._base_time_delta = pd.Timedelta(base_freq)
        assert sample_mode in ["random", "sequent"], "Sample mode=%s is not supported" % sample_mode
        self.sample_mode = sample_mode
        if sample_mode == "sequent":
            assert self._stride is not None
            self._current_datetime = self.begin_time
            self._buffer_mult = 6
            self._buffer_datetime_keys = None
            self._buffer_frame_dat = None
            self._buffer_mask_dat = None
        else:
            self._max_buffer_length = None

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

    @property
    def all_datetime_clips(self):
        if not hasattr(self, "_all_datetime_clips"):
            if self.sample_mode == "sequent":
                stride = self._stride
            else:
                stride = 1
            # remember the current states of the iterator
            sample_mode = self.sample_mode
            if sample_mode == "sequent":
                _current_datetime = self._current_datetime
            else:
                # run one whole epoch in "sequent" mode. reset the mode after finishing.
                self.sample_mode = "sequent"
            self.reset()
            datetime_clips = []
            while not self.use_up:
                datetime_clip = pd.date_range(start=self._current_datetime,
                                              periods=self._seq_len,
                                              freq=self._base_freq)
                if self._is_valid_clip(datetime_clip):
                    datetime_clips.append(datetime_clip)
                    self._current_datetime += stride * self._base_time_delta
                else:
                    self._current_datetime = \
                        self._next_exist_timestamp(timestamp=self._current_datetime)
                    if self._current_datetime is None:
                        # This indicates that there is no timestamp left,
                        # We point the current_datetime to be the next timestamp of self.end_time
                        self._current_datetime = self.end_time + self._base_time_delta
                        break
            self._all_datetime_clips = datetime_clips
            # resume the states
            if sample_mode == "sequent":
                self._current_datetime = _current_datetime
            else:
                self.sample_mode = "random"
        return self._all_datetime_clips

    @property
    def all_avg_int_clips(self):
        if "avg_int" not in self._df.columns:
            raise ValueError("The pandas dataframe does not contain the column 'avg_int'.")
        if not hasattr(self, "_all_avg_int_clips"):
            self._all_avg_int_clips = []
            for datetime_clips in self.all_datetime_clips:
                avg_int_clip = []
                for ele in datetime_clips:
                    if ele in self._df.index:
                        avg_int_clip.append(self._df.loc[ele]["avg_int"])
                    else:
                        avg_int_clip.append(0.0)
                self._all_avg_int_clips.append(np.mean(avg_int_clip))
        return self._all_avg_int_clips

    @staticmethod
    def get_exclude_mask(path=None):
        if path is None:
            path = default_exclude_mask_path
        with np.load(path) as dat:
            exclude_mask = dat['exclude_mask'][:]
            return exclude_mask

    @staticmethod
    def convert_datetime_to_filepath(date_time, file_dir=None):
        """Convert datetime to the filepath

        Parameters
        ----------
        date_time:  datetime.datetime
        file_dir:   str

        Returns
        -------
        ret : str
        """
        ret = os.path.join("%04d" % date_time.year,
                           "%02d" % date_time.month,
                           "%02d" % date_time.day,
                           'RAD%02d%02d%02d%02d%02d00.png'
                           % (date_time.year - 2000, date_time.month, date_time.day,
                              date_time.hour, date_time.minute))
        if file_dir is None:
            file_dir = default_png_file_dir
        ret = os.path.join(file_dir, ret)
        return ret

    @staticmethod
    def convert_datetime_to_maskpath(date_time, file_dir=None):
        """Convert datetime to path of the mask

        Parameters
        ----------
        date_time : datetime.datetime
        file_dir:   str

        Returns
        -------
        ret : str
        """
        ret = os.path.join("%04d" % date_time.year,
                           "%02d" % date_time.month,
                           "%02d" % date_time.day,
                           'RAD%02d%02d%02d%02d%02d00.mask'
                           % (date_time.year - 2000, date_time.month, date_time.day,
                              date_time.hour, date_time.minute))
        if file_dir is None:
            file_dir = default_mask_file_dir
        ret = os.path.join(file_dir, ret)
        return ret

    def set_begin_end(self, begin_ind=None, end_ind=None):
        self._begin_ind = 0 if begin_ind is None else begin_ind
        self._end_ind = self.total_frame_num - 1 if end_ind is None else end_ind

    @property
    def total_frame_num(self):
        return len(self._df)

    @property
    def begin_time(self):
        return self._df.index[self._begin_ind]

    @property
    def end_time(self):
        return self._df.index[self._end_ind]

    @property
    def use_up(self):
        if self.sample_mode == "random":
            return False
        else:
            return self._current_datetime > self.end_time

    def _next_exist_timestamp(self, timestamp):
        next_ind = bisect.bisect_right(self._df.index, timestamp)
        if next_ind >= len(self._df):
            return None
        else:
            return self._df.index[bisect.bisect_right(self._df.index, timestamp)]

    def _is_valid_clip(self, datetime_clip):
        """Check if the given datetime_clip is valid

        Parameters
        ----------
        datetime_clip :

        Returns
        -------
        ret : bool
        """
        missing_count = 0
        for i in range(len(datetime_clip)):
            if datetime_clip[i] not in self._df_index_set:
                missing_count += 1
                if missing_count > self._max_consecutive_missing or \
                        missing_count >= len(datetime_clip):
                    return False
            else:
                missing_count = 0
        return True

    def _load_frames(self, datetime_clips):
        """
        Deprecated. Adapted from the original implementation.
        """
        assert isinstance(datetime_clips, list)
        for clip in datetime_clips:
            assert len(clip) == self._seq_len
        batch_size = len(datetime_clips)
        frame_dat = np.zeros((self._seq_len, batch_size, 1, self._height, self._width),
                             dtype=np.uint8)
        mask_dat = np.zeros((self._seq_len, batch_size, 1, self._height, self._width),
                            dtype=bool)
        if self.sample_mode == "random":
            paths = []
            mask_paths = []
            hit_inds = []
            miss_inds = []
            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set:
                        paths.append(self.convert_datetime_to_filepath(datetime_clips[j][i],
                                                                       file_dir=self.png_file_dir))
                        mask_paths.append(self.convert_datetime_to_maskpath(datetime_clips[j][i],
                                                                            file_dir=self.mask_file_dir))
                        hit_inds.append([i, j])
                    else:
                        miss_inds.append([i, j])
            hit_inds = np.array(hit_inds, dtype=int)
            all_frame_dat = quick_read_frames(path_list=paths,
                                              im_h=self._height,
                                              im_w=self._width,
                                              grayscale=True)
            all_mask_dat = quick_read_masks(mask_paths)
            frame_dat[hit_inds[:, 0], hit_inds[:, 1], :, :, :] = all_frame_dat
            mask_dat[hit_inds[:, 0], hit_inds[:, 1], :, :, :] = all_mask_dat
        else:
            # Get the first_timestamp and the last_timestamp in the datetime_clips
            first_timestamp = datetime_clips[-1][-1]
            last_timestamp = datetime_clips[0][0]
            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set:
                        first_timestamp = min(first_timestamp, timestamp)
                        last_timestamp = max(last_timestamp, timestamp)
            if self._buffer_datetime_keys is None or \
                    not (first_timestamp in self._buffer_datetime_keys
                         and last_timestamp in self._buffer_datetime_keys):
                read_begin_ind = self._df.index.get_loc(first_timestamp)
                read_end_ind = self._df.index.get_loc(last_timestamp) + 1
                read_end_ind = min(read_begin_ind +
                                   self._buffer_mult * (read_end_ind - read_begin_ind),
                                   len(self._df))
                self._buffer_datetime_keys = self._df.index[read_begin_ind:read_end_ind]
                # Fill in the buffer
                paths = []
                mask_paths = []
                for i in range(len(self._buffer_datetime_keys)):
                    paths.append(self.convert_datetime_to_filepath(self._buffer_datetime_keys[i],
                                                                   file_dir=self.png_file_dir))
                    mask_paths.append(self.convert_datetime_to_maskpath(self._buffer_datetime_keys[i],
                                                                        file_dir=self.mask_file_dir))
                self._buffer_frame_dat = quick_read_frames(path_list=paths,
                                                           im_h=self._height,
                                                           im_w=self._width,
                                                           grayscale=True)
                self._buffer_mask_dat = quick_read_masks(mask_paths)
            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set:
                        assert timestamp in self._buffer_datetime_keys
                        ind = self._buffer_datetime_keys.get_loc(timestamp)
                        frame_dat[i, j, :, :, :] = self._buffer_frame_dat[ind, :, :, :]
                        mask_dat[i, j, :, :, :] = self._buffer_mask_dat[ind, :, :, :]
        return frame_dat, mask_dat

    def reset(self, begin_ind=None, end_ind=None):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._current_datetime = self.begin_time

    def random_reset(self):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_ind=np.random.randint(0,
                                                       self.total_frame_num -
                                                       5 * self._seq_len),
                           end_ind=None)
        self._current_datetime = self.begin_time

    def check_new_start(self):
        assert self.sample_mode == "sequent"
        datetime_clip = pd.date_range(start=self._current_datetime,
                                      periods=self._seq_len,
                                      freq=self._base_freq)
        if self._is_valid_clip(datetime_clip):
            return self._current_datetime == self.begin_time
        else:
            return True

    def _sample(self, batch_size, only_return_datetime=False):
        """
        Deprecated. Adapted from the original implementation.

        Sample a minibatch from the hko7 dataset based on the given type and pd_file

        Parameters
        ----------
        batch_size : int
            Batch size
        only_return_datetime : bool
            Whether to only return the datetimes
        Returns
        -------
        frame_dat : np.ndarray
            Shape: (seq_len, valid_batch_size, 1, height, width)
        mask_dat : np.ndarray
            Shape: (seq_len, valid_batch_size, 1, height, width)
        datetime_clips : list
            length should be valid_batch_size
        new_start : bool
        """
        if self.sample_mode == 'sequent':
            if self.use_up:
                raise ValueError("The HKOIterator has been used up!")
            datetime_clips = []
            new_start = False
            for i in range(batch_size):
                while not self.use_up:
                    datetime_clip = pd.date_range(start=self._current_datetime,
                                                  periods=self._seq_len,
                                                  freq=self._base_freq)
                    if self._is_valid_clip(datetime_clip):
                        new_start = new_start or (self._current_datetime == self.begin_time)
                        datetime_clips.append(datetime_clip)
                        self._current_datetime += self._stride * self._base_time_delta
                        break
                    else:
                        new_start = True
                        self._current_datetime = \
                            self._next_exist_timestamp(timestamp=self._current_datetime)
                        if self._current_datetime is None:
                            # This indicates that there is no timestamp left,
                            # We point the current_datetime to be the next timestamp of self.end_time
                            self._current_datetime = self.end_time + self._base_time_delta
                            break
                        continue
            new_start = None if batch_size != 1 else new_start
            if only_return_datetime:
                return datetime_clips, new_start
        else:
            assert only_return_datetime is False
            datetime_clips = []
            new_start = None
            for i in range(batch_size):
                while True:
                    rand_ind = np.random.randint(0, len(self._df), 1)[0]
                    random_datetime = self._df.index[rand_ind]
                    datetime_clip = pd.date_range(start=random_datetime,
                                                  periods=self._seq_len,
                                                  freq=self._base_freq)
                    if self._is_valid_clip(datetime_clip):
                        datetime_clips.append(datetime_clip)
                        break
        frame_dat, mask_dat = self._load_frames(datetime_clips=datetime_clips)
        return frame_dat, mask_dat, datetime_clips, new_start

    def _load_seq(self, datetime_clip):
        # assert isinstance(datetime_clip, list)
        assert len(datetime_clip) == self._seq_len
        frame_dat = np.zeros((self._seq_len, self._height, self._width),
                             dtype=np.uint8)
        mask_dat = np.zeros((self._seq_len, self._height, self._width),
                            dtype=bool)
        if self.sample_mode == "random":
            paths = []
            mask_paths = []
            hit_inds = []
            miss_inds = []
            for i in range(self._seq_len):
                timestamp = datetime_clip[i]
                if timestamp in self._df_index_set:
                    paths.append(self.convert_datetime_to_filepath(datetime_clip[i],
                                                                   file_dir=self.png_file_dir))
                    mask_paths.append(self.convert_datetime_to_maskpath(datetime_clip[i],
                                                                        file_dir=self.mask_file_dir))
                    hit_inds.append(i)
                else:
                    miss_inds.append(i)
            hit_inds = np.array(hit_inds, dtype=int)
            all_frame_dat = quick_read_frames(path_list=paths,
                                              im_h=self._height,
                                              im_w=self._width,
                                              grayscale=True,
                                              layout="NHW")
            all_mask_dat = quick_read_masks(mask_paths, layout="NHW")
            frame_dat[hit_inds, :, :] = all_frame_dat
            mask_dat[hit_inds, :, :] = all_mask_dat
        else:
            # Get the first_timestamp and the last_timestamp in the datetime_clips
            first_timestamp = datetime_clip[-1]
            last_timestamp = datetime_clip[0]
            for i in range(self._seq_len):
                timestamp = datetime_clip[i]
                if timestamp in self._df_index_set:
                    first_timestamp = min(first_timestamp, timestamp)
                    last_timestamp = max(last_timestamp, timestamp)
            if self._buffer_datetime_keys is None or \
                    not (first_timestamp in self._buffer_datetime_keys
                         and last_timestamp in self._buffer_datetime_keys):
                read_begin_ind = self._df.index.get_loc(first_timestamp)
                read_end_ind = self._df.index.get_loc(last_timestamp) + 1
                read_end_ind = min(read_begin_ind +
                                   self._buffer_mult * (read_end_ind - read_begin_ind),
                                   len(self._df))
                self._buffer_datetime_keys = self._df.index[read_begin_ind:read_end_ind]
                # Fill in the buffer
                paths = []
                mask_paths = []
                for i in range(len(self._buffer_datetime_keys)):
                    paths.append(self.convert_datetime_to_filepath(self._buffer_datetime_keys[i],
                                                                   file_dir=self.png_file_dir))
                    mask_paths.append(self.convert_datetime_to_maskpath(self._buffer_datetime_keys[i],
                                                                        file_dir=self.mask_file_dir))
                self._buffer_frame_dat = quick_read_frames(path_list=paths,
                                                           im_h=self._height,
                                                           im_w=self._width,
                                                           grayscale=True,
                                                           layout="NHW")
                self._buffer_mask_dat = quick_read_masks(mask_paths, layout="NHW")
            for i in range(self._seq_len):
                timestamp = datetime_clip[i]
                if timestamp in self._df_index_set:
                    assert timestamp in self._buffer_datetime_keys
                    ind = self._buffer_datetime_keys.get_loc(timestamp)
                    frame_dat[i, :, :] = self._buffer_frame_dat[ind, :, :]
                    mask_dat[i, :, :] = self._buffer_mask_dat[ind, :, :]
        return frame_dat, mask_dat

    def _getitem(self, item, return_datetime=False):
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
        datetime_clip = self.all_datetime_clips[item]
        frame_dat, mask_dat = self._load_seq(datetime_clip=datetime_clip)
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
        if self.norm_mode == "01":
            frame_dat = frame_dat.float() / 255.0
            mask_dat = mask_dat.float()
        else:
            raise NotImplementedError
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
        if return_datetime:
            return frame_dat, mask_dat, datetime_clip
        else:
            return frame_dat, mask_dat

    def __getitem__(self, item):
        return self._getitem(item=item, return_datetime=False)

    def __len__(self):
        return len(self.all_datetime_clips)
