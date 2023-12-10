r"""Code is adapted from https://github.com/Lightning-AI/torchmetrics/blob/54a06013cdac4895bf8e85b583c5f220388ebc1d/src/torchmetrics/image/fid.py#L127-L300"""
from copy import deepcopy
from typing import Any, List, Optional, Union
import math
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, functional as F
from einops import rearrange

from torchmetrics.metric import Metric
from torchmetrics.image.fid import _compute_fid
# if _TORCH_FIDELITY_AVAILABLE:
#     from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
# else:
#
#     class FeatureExtractorInceptionV3(Module):  # type: ignore
#         pass
#
#     __doctest_skip__ = ["FrechetInceptionDistance", "FID"]
from .download import load_i3d_pretrained
from ...utils.optim import disable_train


class I3DWrapper(Module):

    def __init__(self, channels=400):
        super().__init__()
        self.channels = channels
        self.i3d = load_i3d_pretrained(channels=channels, device=torch.device("cpu"))

    @staticmethod
    def preprocess(video, target_resolution=224):
        r"""
        Parameters
        ----------
        video:  torch.Tensor
            shape = (b, t, 3, h, w)
            value range fomr 0 to 1
        target_resolution:  int
            224 by default

        Returns
        -------

        """
        b, t, c, h, w = video.shape
        # scale shorter side to resolution
        scale = target_resolution / min(h, w)
        if h < w:
            target_size = (target_resolution, math.ceil(w * scale))
        else:
            target_size = (math.ceil(h * scale), target_resolution)
        video = rearrange(video, "b t c h w -> (b t) c h w")
        video = F.interpolate(video, size=target_size, mode='bilinear',
                              align_corners=False)
        # center crop
        _, _, h, w = video.shape
        w_start = (w - target_resolution) // 2
        h_start = (h - target_resolution) // 2
        video = video[:, :, h_start:h_start + target_resolution, w_start:w_start + target_resolution]
        video = rearrange(video, "(b t) c h w -> b c t h w", b=b, t=t).contiguous()  # CTHW

        video -= 0.5
        return video * 2  # value range from -1 to 1

    def forward(self, video):
        r"""
        Parameters
        ----------
        video:  torch.Tensor
            shape = (b, t, c, h, w)
            value from 0 to 1

        Returns
        -------
        logits: torch.Tensor
            shape = (b, self.channels)
        """
        processed_video = self.preprocess(video=video)
        return self.i3d(processed_video)


class FrechetVideoDistance(Metric):
    r"""Calculates Fréchet video distance (FVD) which is used to access the quality of generated images. Given
    by.

    .. math::
        FVD = |\mu - \mu_w| + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``videos`` (:class:`~torch.Tensor`): tensor with videos feed to the feature extractor with
    - ``real`` (:class:`~bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``fvd`` (:class:`~torch.Tensor`): float scalar tensor with mean FVD value over samples

    Args:
        feature:
            Either an integer or ``nn.Module``:

            - an integer will indicate the i3d feature layer to choose. Can be one of the following:
              400, 600
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` not in [400, 600]
        TypeError:
            If ``feature`` is not an ``str``, ``int`` or ``torch.nn.Module``
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example:
        >>> _ = torch.manual_seed(123)
        >>> fvd = FrechetVideoDistance(feature=400)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 10, 3, 224, 224), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 10, 3, 224, 224), dtype=torch.uint8)
        >>> fvd.update(imgs_dist1, real=True)
        >>> fvd.update(imgs_dist2, real=False)
        >>> fvd.compute()
        tensor(12.7202)
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor
    
    default_layout = "NTCHW"
    einops_default_layout = "N T C H W"
    default_t_axis = 1
    min_t = 9

    def __init__(
        self,
        feature: Union[int, Module] = 400,
        layout: str = "NTCHW",
        reset_real_features: bool = True,
        normalize: bool = False,
        auto_t: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.layout = layout

        if isinstance(feature, int):
            num_features = feature
            # if not _TORCH_FIDELITY_AVAILABLE:
            #     raise ModuleNotFoundError(
            #         "FrechetInceptionDistance metric requires that `Torch-fidelity` is installed."
            #         " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
            #     )
            valid_int_input = [400, 600]
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception = I3DWrapper(channels=feature)

        elif isinstance(feature, Module):
            self.inception = feature
            dummy_image = torch.randint(0, 255, (1, 9, 3, 299, 299), dtype=torch.uint8)
            num_features = self.inception(dummy_image).shape[-1]
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        self.auto_t = auto_t

        mx_nb_feets = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_nb_feets).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_nb_feets).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        disable_train(self)

    @property
    def einops_layout(self):
        if not hasattr(self, "_einops_layout"):
            self._einops_layout = " ".join(self.layout)
        return self._einops_layout

    def update(self, videos: Tensor, real: bool) -> None:  # type: ignore
        r"""
        Update the state with extracted features.

        Parameters
        ----------
        videos: torch.Tensor
            shape = (b, t, c, h, w), t >= 9, c = 3 or 1
            value from 0 to 255 if self.normalize else 0 to 1
        real:   bool
        """
        videos = rearrange(videos, f"{self.einops_layout} -> {self.einops_default_layout}")
        if videos.shape[1] < self.min_t:
            if self.auto_t:
                videos = torch.repeat_interleave(videos, repeats=2, dim=self.default_t_axis)
            else:
                raise ValueError(f"The temporal length of the input is smaller than the minimal requirement:"
                                 f" videos.shape[1] = {videos.shape[1]} < {self.min_t}.")
        videos = videos / 255.0 if self.normalize else videos
        c = videos.shape[2]
        if c == 1:  # see discussion:https://github.com/google/compare_gan/issues/13 and reference implementation: https://github.com/google/compare_gan/blob/560697ee213f91048c6b4231ab79fcdd9bf20381/compare_gan/src/eval_gan_lib.py#L786-L791
            videos = videos.repeat(1, 1, 3, 1, 1)
        features = self.inception(videos)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += videos.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += videos.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    def reset(self) -> None:
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()
