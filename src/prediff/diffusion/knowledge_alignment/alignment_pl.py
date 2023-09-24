"""
Code is adapted from https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py
"""
import warnings
from typing import Sequence, Union, Dict, Any, Optional, Callable
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from diffusers.models.autoencoder_kl import AutoencoderKLOutput, DecoderOutput

from ...utils.distributions import DiagonalGaussianDistribution
from ..utils import make_beta_schedule, extract_into_tensor, default
from ...utils.layout import parse_layout_shape
from ...utils.optim import disabled_train, get_loss_fn


class AlignmentPL(pl.LightningModule):

    def __init__(self,
                 torch_nn_module: nn.Module,
                 target_fn: Callable,
                 layout: str = "NTHWC",
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type: str = "l2",
                 monitor="val_loss",
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 # latent diffusion
                 first_stage_model: Union[Dict[str, Any], nn.Module] = None,
                 cond_stage_model: Union[str, Dict[str, Any], nn.Module] = None,
                 num_timesteps_cond=None,
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 scale_by_std=False,
                 scale_factor=1.0,):
        r"""
        Parameters
        ----------
        Parameters
        ----------
        torch_nn_module:  nn.Module
            The `.forward()` method of model should have the following signature:
            `output = model.forward(zt, t, y, zc, **kwargs)`
        target_fn:  Callable
            The function that the `torch_nn_module` is going to learn.
            The signature of `target_fn` should be:
            `violation_score = target_fn(x, y=None, **kwargs)`
        layout: str
            e.g., "NTHWC", "NHWC".
        timesteps:  int
            1000 by default.
        beta_schedule:  str
            one of ["linear", "cosine", "sqrt_linear", "sqrt"].
        loss_type:  str
            one of ["l2", "l1"].
        monitor:    str
            name of logged var for selecting best val model.
        linear_start:   float
        linear_end:     float
        cosine_s:       float
        given_betas:    Optional
            If provided, `linear_start`, `linear_end`, `cosine_s` take no effect.
            If None, `linear_start`, `linear_end`, `cosine_s` are used to generate betas via `make_beta_schedule()`.
        first_stage_model:  Dict or nn.Module
            Dict        : configs for instantiating the first_stage_model.
            nn.Module   : a model that has method ".encode()" to encode the inputs.
        cond_stage_model:   str or Dict or nn.Module
            "__is_first_stage__": use the first_stage_model also for encoding conditionings.
            Dict                : configs for instantiating the cond_stage_model.
            nn.Module           : a model that has method ".encode()" or use `self()` to encodes the conditionings.
        cond_stage_trainable:   bool
            Whether to train the cond_stage_model jointly
        num_timesteps_cond: int
        cond_stage_forward: str
            The name of the forward method of the cond_stage_model.
        scale_by_std
        scale_factor
        """
        super(AlignmentPL, self).__init__()
        self.torch_nn_module = torch_nn_module
        self.target_fn = target_fn
        self.loss_fn = get_loss_fn(loss_type)
        self.layout = layout
        self.parse_layout_shape(layout=layout)

        if monitor is not None:
            self.monitor = monitor

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        assert self.num_timesteps_cond <= timesteps
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

        self.cond_stage_trainable = cond_stage_trainable
        self.scale_by_std = scale_by_std
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_model)
        self.instantiate_cond_stage(cond_stage_model, cond_stage_forward)

    def parse_layout_shape(self, layout):
        parsed_dict = parse_layout_shape(layout=layout)
        self.batch_axis = parsed_dict["batch_axis"]
        self.t_axis = parsed_dict["t_axis"]
        self.h_axis = parsed_dict["h_axis"]
        self.w_axis = parsed_dict["w_axis"]
        self.c_axis = parsed_dict["c_axis"]
        self.all_slice = [slice(None, None), ] * len(layout)

    def extract_into_tensor(self, a, t, x_shape):
        return extract_into_tensor(a=a, t=t, x_shape=x_shape,
                                   batch_axis=self.batch_axis)

    @property
    def loss_mean_dim(self):
        # mean over all dims except for batch_axis.
        if not hasattr(self, "_loss_mean_dim"):
            _loss_mean_dim = list(range(len(self.layout)))
            _loss_mean_dim.pop(self.batch_axis)
            self._loss_mean_dim = tuple(_loss_mean_dim)
        return self._loss_mean_dim

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def make_cond_schedule(self, ):
        cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        cond_ids[:self.num_timesteps_cond] = ids
        self.register_buffer('cond_ids', cond_ids)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        # TODO: restarted_from_ckpt not configured
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x, _ = self.get_input(batch)
            x = x.to(self.device)
            x = rearrange(x, f"{self.einops_layout} -> {self.einops_spatial_layout}")
            z = self.encode_first_stage(x)
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def instantiate_first_stage(self, first_stage_model):
        if isinstance(first_stage_model, nn.Module):
            model = first_stage_model
        else:
            assert first_stage_model is None
            raise NotImplementedError("No default first_stage_model supported yet!")
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, cond_stage_model, cond_stage_forward):
        if cond_stage_model is None:
            self.cond_stage_model = None
            self.cond_stage_forward = None
            return

        is_first_stage_flag = cond_stage_model == "__is_first_stage__"
        if cond_stage_model == "__is_first_stage__":
            model = self.first_stage_model
            if self.cond_stage_trainable:
                warnings.warn("`cond_stage_trainable` is True while `cond_stage_model` is '__is_first_stage__'. "
                              "force `cond_stage_trainable` to be False")
                self.cond_stage_trainable = False
        elif isinstance(cond_stage_model, nn.Module):
            model = cond_stage_model
        else:
            raise NotImplementedError
        self.cond_stage_model = model
        if (self.cond_stage_model is not None) and (not self.cond_stage_trainable):
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False

        if cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                cond_stage_forward = self.cond_stage_model.encode
            else:
                cond_stage_forward = self.cond_stage_model.__call__
        else:
            assert hasattr(self.cond_stage_model, cond_stage_forward)
            cond_stage_forward = getattr(self.cond_stage_model, cond_stage_forward)

        def wrapper(cond_stage_forward: Callable, is_first_stage_flag=False):
            def func(c: Dict[str, Any]):
                if is_first_stage_flag:
                    # in this case, `cond_stage_model` is equivalent to `self.first_stage_model`,
                    # which takes `torch.Tensor` instead of `Dict` as input.
                    c = c.get("y")  # get the conditioning tensor
                    batch_size = c.shape[self.batch_axis]
                    c = rearrange(c, f"{self.einops_layout} -> {self.einops_spatial_layout}")
                c = cond_stage_forward(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
                elif isinstance(c, AutoencoderKLOutput):
                    c = c.latent_dist.mode()
                else:
                    pass
                if is_first_stage_flag:
                    c = rearrange(c, f"{self.einops_spatial_layout} -> {self.einops_layout}", N=batch_size)
                return c
            return func
        self.cond_stage_forward = wrapper(cond_stage_forward, is_first_stage_flag)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        elif isinstance(encoder_posterior, AutoencoderKLOutput):
            z = encoder_posterior.latent_dist.sample()
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        r"""
        Try the following approaches to encode the conditional input `c`:
        1. `self.cond_stage_forward` is a str, call the method of `self.cond_stage_model`.
        2. call `encode()` method of `self.cond_stage_model`.
        3. call `forward()` of `self.cond_stage_model`, i.e., `self.cond_stage_model()`.
        """
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @property
    def einops_layout(self):
        return " ".join(self.layout)

    @property
    def einops_spatial_layout(self):
        if not hasattr(self, "_einops_spatial_layout"):
            assert len(self.layout) == 4 or len(self.layout) == 5
            self._einops_spatial_layout = "(N T) C H W" if self.layout.find("T") else "N C H W"
        return self._einops_spatial_layout

    @torch.no_grad()
    def decode_first_stage(self, z, force_not_quantize=False):
        z = 1. / self.scale_factor * z
        batch_size = z.shape[self.batch_axis]
        z = rearrange(z, f"{self.einops_layout} -> {self.einops_spatial_layout}")
        output = self.first_stage_model.decode(z)
        if isinstance(output, DecoderOutput):
            output = output.sample
        output = rearrange(output, f"{self.einops_spatial_layout} -> {self.einops_layout}", N=batch_size)
        return output

    @torch.no_grad()
    def encode_first_stage(self, x):
        encoder_posterior = self.first_stage_model.encode(x)
        output = self.get_first_stage_encoding(encoder_posterior).detach()
        return output

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (self.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    @torch.no_grad()
    def get_input(self, batch, **kwargs):
        r"""
        dataset dependent
        re-implement it for each specific dataset

        Parameters
        ----------
        batch:  Any
            raw data batch from specific dataloader

        Returns
        -------
        out:    Sequence[torch.Tensor, Dict[str, Any]]
            out[0] should be a torch.Tensor which is the target to generate
            out[1] should be a dict consists of several key-value pairs for conditioning
        """
        return batch

    def forward(self, batch, t=None, verbose=False, return_verbose=False, **kwargs):
        # similar to latent diffusion
        x, c, aux_input_dict = self.get_input(batch)  # torch.Tensor, Dict[str, Any], Dict[str, Any]
        if verbose:
            print("inputs:")
            print(f"x.shape = {x.shape}")
            for key, val in c.items():
                if hasattr(val, "shape"):
                    print(f"{key}.shape = {val.shape}")
        batch_size = x.shape[self.batch_axis]
        x = x.to(self.device)
        x_spatial = rearrange(x, f"{self.einops_layout} -> {self.einops_spatial_layout}")
        z = self.encode_first_stage(x_spatial)
        if verbose:
            print("after first stage:")
            print(f"z.shape = {z.shape}")
        # xrec = self.decode_first_stage(z)
        z = rearrange(z, f"{self.einops_spatial_layout} -> {self.einops_layout}", N=batch_size)

        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        y = c if isinstance(c, torch.Tensor) else c.get("y", None)
        if self.cond_stage_model is not None:
            assert c is not None
            zc = self.cond_stage_forward(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t]
                zc = self.q_sample(x_start=zc, t=tc, noise=torch.randn_like(c.float()))
            if verbose and hasattr(zc, "shape"):
                print(f"zc.shape = {zc.shape}")
        else:
            zc = y
        if verbose and hasattr(y, "shape"):
            print(f"y.shape = {y.shape}")
        # calculate the loss
        zt = self.q_sample(x_start=z, t=t, noise=torch.randn_like(z))
        target = self.target_fn(x, y, **aux_input_dict)
        pred = self.torch_nn_module(zt, t, y=y, zc=zc, **aux_input_dict)
        loss = self.loss_fn(pred, target)
        # other metrics
        with torch.no_grad():
            mae = F.l1_loss(pred, target).float().cpu().item()
            avg_gt = torch.abs(target).mean().float().cpu().item()
        loss_dict = {
            "mae": mae,
            "avg_gt": avg_gt,
            "relative_mae": mae / (avg_gt + 1E-8),
        }
        if return_verbose:
            return loss, loss_dict, \
                   {"pred": pred, "target": target, "t": t, "zc": zc, "zt": zt}
        else:
            return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        loss_dict = {f"train/{key}": val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        loss_dict = {f"val/{key}": val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        loss_dict = {f"test/{key}": val for key, val in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.torch_nn_module.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


def get_sample_align_fn(sample_align_model):
    r"""
    Code is adapted from https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/scripts/classifier_sample.py#L54-L61
    """
    def sample_align_fn(x, *args, **kwargs):
        r"""
        Calculates `grad(log(p(y|x)))`
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).

        Parameters
        ----------
        x:  torch.Tensor

        Returns
        -------
        grad
        """
        # with torch.inference_mode(False):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = sample_align_model(x_in, *args, **kwargs)
            grad = torch.autograd.grad(logits.sum(), x_in, allow_unused=True)[0]
            return grad
    return sample_align_fn
