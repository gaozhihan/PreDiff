"""Code is adapted from https://github.com/MIT-AI-Accelerator/neurips-2020-sevir. Their license is MIT License."""
from typing import Optional, Sequence
import re
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange
from torchmetrics import Metric
from .sevir_dataloader import SEVIRDataLoader


def _threshold(target, pred, T):
    """
    Returns binary tensors t,p the same shape as target & pred.  t = 1 wherever
    target > t.  p =1 wherever pred > t.  p and t are set to 0 wherever EITHER
    t or p are nan.
    This is useful for counts that don't involve correct rejections.

    Parameters
    ----------
    target
        torch.Tensor
    pred
        torch.Tensor
    T
        numeric_type:   threshold
    Returns
    -------
    t
    p
    """
    t = (target >= T).float()
    p = (pred >= T).float()
    is_nan = torch.logical_or(torch.isnan(target),
                              torch.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p


def _calc_hits_misses_fas(t, p):
    hits = torch.sum(t * p)
    misses = torch.sum(t * (1 - p))
    fas = torch.sum((1 - t) * p)
    return hits, misses, fas


def _pod(target, pred, T, eps=1e-6):
    """
    Single channel version of probability_of_detection
    """
    t, p = _threshold(target, pred, T)
    hits, misses, fas = _calc_hits_misses_fas(t, p)
    # return (hits + eps) / (hits + misses + eps)
    return hits / (hits + misses + eps)


def _sucr(target, pred, T, eps=1e-6):
    """
    Single channel version of success_rate
    """
    t, p = _threshold(target, pred, T)
    hits, misses, fas = _calc_hits_misses_fas(t, p)
    # return (hits + eps) / (hits + fas + eps)
    return hits / (hits + fas + eps)


def _csi(target, pred, T, eps=1e-6):
    """
    Single channel version of csi
    """
    t, p = _threshold(target, pred, T)
    hits, misses, fas = _calc_hits_misses_fas(t, p)
    # return (hits + eps) / (hits + misses + fas + eps)
    return hits / (hits + misses + fas + eps)


def _bias(target, pred, T, eps=1e-6):
    """
    Single channel version of csi
    """
    t, p = _threshold(target, pred, T)
    hits, misses, fas = _calc_hits_misses_fas(t, p)
    # return (hits + fas + eps) / (hits + misses + eps)
    return (hits + fas) / (hits + misses + eps)


class SEVIRSkillScore(Metric):
    r"""
    The calculation of skill scores in SEVIR challenge is slightly different:
        `mCSI = sum(mCSI_t) / T`
    See https://github.com/MIT-AI-Accelerator/sevir_challenges/blob/dev/radar_nowcasting/RadarNowcastBenchmarks.ipynb for more details.
    """
    full_state_update: bool = True

    def __init__(self,
                 layout: str = "NHWT",
                 mode: str = "0",
                 seq_len: Optional[int] = None,
                 preprocess_type: str = "sevir",
                 threshold_list: Sequence[int] = (16, 74, 133, 160, 181, 219),
                 metrics_list: Sequence[str] = ("csi", "bias", "sucr", "pod"),
                 eps: float = 1e-4,
                 ):
        r"""
        Parameters
        ----------
        seq_len
        layout
        mode:   str
            Should be in ("0", "1", "2")
            "0":
                cumulates hits/misses/fas of all test pixels
                score_avg takes average over all thresholds
                return
                    score_thresh shape = (1, )
                    score_avg shape = (1, )
            "1":
                cumulates hits/misses/fas of each step
                score_avg takes average over all thresholds while keeps the seq_len dim
                return
                    score_thresh shape = (seq_len, )
                    score_avg shape = (seq_len, )
            "2":
                cumulates hits/misses/fas of each step
                score_avg takes average over all thresholds, then takes average over the seq_len dim
                return
                    score_thresh shape = (1, )
                    score_avg shape = (1, )
        preprocess_type：    str
            valid values are "sevir", f"sevir_pool{x}", where `x` is the pool scale factor
        threshold_list
        """
        super().__init__()
        self.layout = layout
        self.parse_layout()
        assert preprocess_type == "sevir" or preprocess_type.startswith("sevir_pool")
        self.preprocess_type = preprocess_type
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps
        self.mode = mode
        self.seq_len = seq_len
        if mode in ("0", ):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list), )
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)

        else:
            raise NotImplementedError(f"mode {mode} not supported!")

        self.add_state("hits",
                       default=torch.zeros(state_shape),
                       dist_reduce_fx="sum")
        self.add_state("misses",
                       default=torch.zeros(state_shape),
                       dist_reduce_fx="sum")
        self.add_state("fas",
                       default=torch.zeros(state_shape),
                       dist_reduce_fx="sum")

    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find('T')
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims

    def parse_layout(self):
        self.batch_axis = self.layout.find("N")
        self.einops_layout = " ".join(self.layout)
        self.einops_spatial_layout = "(N T) C H W"

    @staticmethod
    def pod(hits, misses, fas, eps):
        return hits / (hits + misses + eps)

    @staticmethod
    def sucr(hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    @staticmethod
    def csi(hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)

    @staticmethod
    def bias(hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias

    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        """
        Parameters
        ----------
        pred, target:   torch.Tensor
        threshold:  int

        Returns
        -------
        hits, misses, fas:  torch.Tensor
            each has shape (seq_len, )
        """
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = torch.sum(t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum((1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas

    def preprocess(self, pred, target):
        if self.preprocess_type == "sevir":
            pred = SEVIRDataLoader.process_data_dict_back(
                data_dict={'vil': pred.detach().float()})['vil']
            target = SEVIRDataLoader.process_data_dict_back(
                data_dict={'vil': target.detach().float()})['vil']
        elif self.preprocess_type.startswith("sevir_pool"):
            pred = SEVIRDataLoader.process_data_dict_back(
                data_dict={'vil': pred.detach().float()})['vil']
            target = SEVIRDataLoader.process_data_dict_back(
                data_dict={'vil': target.detach().float()})['vil']
            self.pool_scale = int(re.search(r'\d+', self.preprocess_type).group())
            batch_size = target.shape[0]
            pred = rearrange(pred, f"{self.einops_layout} -> {self.einops_spatial_layout}")
            target = rearrange(target, f"{self.einops_layout} -> {self.einops_spatial_layout}")
            pred = F.max_pool2d(pred, kernel_size=self.pool_scale, stride=self.pool_scale)
            target = F.max_pool2d(target, kernel_size=self.pool_scale, stride=self.pool_scale)
            pred = rearrange(pred, f"{self.einops_spatial_layout} -> {self.einops_layout}", N=batch_size)
            target = rearrange(target, f"{self.einops_spatial_layout} -> {self.einops_layout}", N=batch_size)
        else:
            raise NotImplementedError
        return pred, target

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_seq_hits_misses_fas(pred, target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas

    def compute(self):
        metrics_dict = {
            'pod': self.pod,
            'csi': self.csi,
            'sucr': self.sucr,
            'bias': self.bias}
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}
        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len, ))
            else:
                score_avg = 0
            # shape = (len(threshold_list), seq_len) if self.keep_seq_len_dim,
            # else shape = (len(threshold_list),)
            scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2", ):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret
