"""
Code is adapted from https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py
"""
import warnings
from typing import Sequence, Union, Dict, Any, Optional, Callable
from copy import deepcopy
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from diffusers.models.autoencoder_kl import AutoencoderKLOutput, DecoderOutput

from ..utils.ema import LitEma
from ..utils.distributions import DiagonalGaussianDistribution
from .utils import make_beta_schedule, extract_into_tensor, noise_like, default
from ..utils.layout import parse_layout_shape
from ..utils.optim import disabled_train


class LatentDiffusion(pl.LightningModule):

    def __init__(self,
                 torch_nn_module: nn.Module,
                 layout: str = "NTHWC",
                 data_shape: Sequence[int] = (10, 128, 128, 4),
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 monitor="val/loss",
                 use_ema=True,
                 log_every_t=100,
                 clip_denoised=False,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,
                 l_simple_weight=1.,
                 parameterization="eps",
                 learn_logvar=False,
                 logvar_init=0.,
                 # latent diffusion
                 latent_shape: Sequence[int] = (10, 16, 16, 4),
                 first_stage_model: nn.Module = None,
                 cond_stage_model: Union[str, nn.Module] = None,
                 num_timesteps_cond=None,
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 scale_by_std=False,
                 scale_factor=1.0,
                 ):
        r"""
        Parameters
        ----------

        torch_nn_module:  nn.Module
            The `.forward()` method of model should have the following signature:
            `x_hat = model.forward(x, t, *args, **kwargs)`
        layout: str
            e.g., "NTHWC", "NHWC".
        data_shape: Sequence[int]
            The shape of each data entry. Corresponds to `layout` without the batch axis "N".
        timesteps:  int
            1000 by default.
        beta_schedule:  str
            one of ["linear", "cosine", "sqrt_linear", "sqrt"].
        loss_type:  str
            one of ["l2", "l1"].
        monitor:    str
            name of logged var for selecting best val model.
        use_ema:    bool
        log_every_t:    int
            log intermediate denoising steps. Should be <= `timesteps`.
        clip_denoised:  bool
        linear_start:   float
        linear_end:     float
        cosine_s:       float
        given_betas:    Optional
            If provided, `linear_start`, `linear_end`, `cosine_s` take no effect.
            If None, `linear_start`, `linear_end`, `cosine_s` are used to generate betas via `make_beta_schedule()`.
        original_elbo_weight:   float
            0. by default
        v_posterior:  float
            0. by default
            weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight:    float
            1. by default
        parameterization:   str
            "eps" by default, to predict the noise from `t` to `t-1`.
            "x0" to predict the `x_{t-1}` from `x_t`.
            all assuming fixed variance schedules.
        learn_logvar:   bool
            use fixed var by default.
        logvar_init:    float
            (initial) values of `logvar`.

        latent_shape:   Sequence[int]
            The shape of downsampled data entry. Corresponds to `layout` without the batch axis "N".
        first_stage_model:  nn.Module
            nn.Module   : a model that has method ".encode()" to encode the inputs.
        cond_stage_model:   str or nn.Module
            "__is_first_stage__": use the first_stage_model also for encoding conditionings.
            nn.Module           : a model that has method ".encode()" or use `self()` to encodes the conditionings.
        cond_stage_trainable:   bool
            Whether to train the cond_stage_model jointly
        num_timesteps_cond: int
        cond_stage_forward: str
            The name of the forward method of the cond_stage_model.
        scale_by_std
        scale_factor
        """
        super(LatentDiffusion, self).__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.torch_nn_module = torch_nn_module
        self.layout = layout
        self.data_shape = data_shape
        self.parse_layout_shape(layout=layout)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.torch_nn_module)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(logvar, requires_grad=True)
        else:
            self.register_buffer('logvar', logvar)

        self.latent_shape = latent_shape
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

    def set_alignment(self, alignment_fn: Callable = None):
        r"""
        Call this method to set alignment after __init__ of LatentDiffusion,
        to avoid error "cannot assign module before Module.__init__() call"
        when assigning alignment model to the LatentDiffusion before its __init__.

        Parameters
        ----------
        alignment_fn: Callable
            Should have signature `alignment_fn(zt, t, zc=None, y=None, xt=None, **kwargs)`.
        """
        self.alignment_fn = alignment_fn

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

    def get_batch_data_shape(self, batch_size=1):
        if not hasattr(self, "batch_data_shape"):  # `self.batch_data_shape` not set
            _batch_data_shape = deepcopy(list(self.data_shape))
            _batch_data_shape.insert(self.batch_axis, batch_size)
        elif self.batch_data_shape[self.batch_axis] != batch_size:  # `batch_size` is changed
            _batch_data_shape = deepcopy(list(self.batch_data_shape))
            _batch_data_shape[self.batch_axis] = batch_size
        else:
            return self.batch_data_shape
        self.batch_data_shape = tuple(_batch_data_shape)
        return self.batch_data_shape

    def get_batch_latent_shape(self, batch_size=1):
        if not hasattr(self, "batch_latent_shape"):  # `self.batch_latent_shape` not set
            _batch_latent_shape = deepcopy(list(self.latent_shape))
            _batch_latent_shape.insert(self.batch_axis, batch_size)
        elif self.batch_latent_shape[self.batch_axis] != batch_size:  # `batch_size` is changed
            _batch_latent_shape = deepcopy(list(self.batch_latent_shape))
            _batch_latent_shape[self.batch_axis] = batch_size
        else:
            return self.batch_latent_shape
        self.batch_latent_shape = tuple(_batch_latent_shape)
        return self.batch_latent_shape

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

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.torch_nn_module.parameters())
            self.model_ema.copy_to(self.torch_nn_module)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.torch_nn_module.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

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

    @property
    def einops_layout(self):
        return " ".join(self.layout)

    @property
    def einops_spatial_layout(self):
        if not hasattr(self, "_einops_spatial_layout"):
            assert len(self.layout) == 4 or len(self.layout) == 5
            self._einops_spatial_layout =  "(N T) C H W" if self.layout.find("T") else "N C H W"
        return self._einops_spatial_layout

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

    @torch.no_grad()
    def decode_first_stage(self, z):
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

    def apply_model(self, x_noisy, t, cond):
        x_recon = self.torch_nn_module(x_noisy, t, cond)
        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def forward(self, batch, verbose=False):
        x, c = self.get_input(batch)  # torch.Tensor, Dict[str, Any]
        if verbose:
            print("inputs:")
            print(f"x.shape = {x.shape}")
            for key, val in c.items():
                if hasattr(val, "shape"):
                    print(f"{key}.shape = {val.shape}")
        batch_size = x.shape[self.batch_axis]
        x = x.to(self.device)
        x = rearrange(x, f"{self.einops_layout} -> {self.einops_spatial_layout}")
        z = self.encode_first_stage(x)
        if verbose:
            print("after first stage:")
            print(f"z.shape = {z.shape}")
        # xrec = self.decode_first_stage(z)
        z = rearrange(z, f"{self.einops_spatial_layout} -> {self.einops_layout}", N=batch_size)

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        if self.cond_stage_model is not None:
            assert c is not None
            zc = self.cond_stage_forward(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t]
                zc = self.q_sample(x_start=zc, t=tc, noise=torch.randn_like(c.float()))
            if verbose and hasattr(zc, "shape"):
                print(f"zc.shape = {zc.shape}")
        else:
            zc = c if isinstance(c, torch.Tensor) else c.get("y", None)
        return self.p_losses(z, zc, t, noise=None)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.torch_nn_module)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self(batch)
        with self.ema_scope():
            _, loss_dict_ema = self(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (self.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self.extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # TODO: add v-prediction
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=self.loss_mean_dim)
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=self.loss_mean_dim)
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self.extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self.extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, zt, zc, t, clip_denoised: bool,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(zt, t_in, zc)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, zt, t, zc, **corrector_kwargs)

        if self.parameterization == "eps":
            z_recon = self.predict_start_from_noise(zt, t=t, noise=model_out)
        elif self.parameterization == "x0":
            z_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            z_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=z_recon, x_t=zt, t=t)
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, z_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def aligned_mean(self, zt, t, zc, y,
                     orig_mean, orig_log_var, **kwargs):
        align_gradient = self.alignment_fn(zt, t, zc=zc, y=y, **kwargs)
        new_mean = orig_mean - (0.5 * orig_log_var).exp() * align_gradient
        return new_mean

    @torch.no_grad()
    def p_sample(self, zt, zc, t, y=None, use_alignment=False, alignment_kwargs=None,
                 clip_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        batch_size = zt.shape[self.batch_axis]
        device = zt.device
        outputs = self.p_mean_variance(zt=zt, zc=zc, t=t, clip_denoised=clip_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if use_alignment:
            if alignment_kwargs is None:
                alignment_kwargs = {}
            model_mean, posterior_variance, model_log_variance, *_ = outputs
            model_mean = self.aligned_mean(zt=zt, t=t, zc=zc, y=y,
                                           orig_mean=model_mean, orig_log_var=model_log_variance,
                                           **alignment_kwargs)
            outputs = (model_mean, posterior_variance, model_log_variance, *outputs[3:])
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(zt.shape, device) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask_shape = [1, ] * len(zt.shape)
        nonzero_mask_shape[self.batch_axis] = batch_size
        nonzero_mask = (1 - (t == 0).float()).reshape(*nonzero_mask_shape)

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, y=None,
                      use_alignment=False, alignment_kwargs=None,
                      return_intermediates=False, x_T=None,
                      verbose=False, callback=None, timesteps=None,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        batch_size = shape[self.batch_axis]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) \
            if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(zt=img, zc=cond, t=ts, y=y,
                                use_alignment=use_alignment,
                                alignment_kwargs=alignment_kwargs,
                                clip_denoised=self.clip_denoised, )
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16,
               use_alignment=False, alignment_kwargs=None,
               return_intermediates=False, x_T=None,
               verbose=False, timesteps=None,
               mask=None, x0=None, shape=None, return_decoded=True, **kwargs):
        if use_alignment:
            assert self.alignment_fn is not None, "Alignment function not set."
        if shape is None:
            shape = self.get_batch_latent_shape(batch_size=batch_size)
        if self.cond_stage_model is not None:
            assert cond is not None
            cond_tensor_slice = [slice(None, None), ] * len(self.data_shape)
            cond_tensor_slice[self.batch_axis] = slice(0, batch_size)
            if isinstance(cond, dict):
                zc = {key: cond[key][cond_tensor_slice] if not isinstance(cond[key], list) else
                list(map(lambda x: x[cond_tensor_slice], cond[key])) for key in cond}
            else:
                zc = [c[cond_tensor_slice] for c in cond] if isinstance(cond, list) else cond[cond_tensor_slice]
            zc = self.cond_stage_forward(zc)
        else:
            zc = cond if isinstance(cond, torch.Tensor) else cond.get("y", None)
        y = cond if isinstance(cond, torch.Tensor) else cond.get("y", None)
        output = self.p_sample_loop(
            cond=zc, shape=shape, y=y,
            use_alignment=use_alignment, alignment_kwargs=alignment_kwargs,
            return_intermediates=return_intermediates, x_T=x_T,
            verbose=verbose, timesteps=timesteps,
            mask=mask, x0=x0)

        if return_decoded:
            if return_intermediates:
                samples, intermediates = output
                decoded_samples = self.decode_first_stage(samples)
                decoded_intermediates = [self.decode_first_stage(ele) for ele in intermediates]
                output = [decoded_samples, decoded_intermediates]
            else:
                output = self.decode_first_stage(output)
        return output

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.torch_nn_module.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
