import warnings
from typing import Sequence, Union, Dict
from shutil import copyfile
import inspect
from collections import OrderedDict
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torchmetrics
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    Callback, LearningRateMonitor, DeviceStatsMonitor,
    EarlyStopping, ModelCheckpoint, )
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange

from prediff.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
from prediff.datasets.sevir.visualization import vis_sevir_seq
from prediff.datasets.sevir.evaluation import SEVIRSkillScore
from prediff.evaluation.fvd import FrechetVideoDistance
from prediff.utils.pl_checkpoint import pl_load
from prediff.utils.download import (
    download_pretrained_weights,
    pretrained_sevirlr_vae_name,
    pretrained_sevirlr_earthformerunet_name,
    pretrained_sevirlr_alignment_name)
from prediff.utils.optim import warmup_lambda, disable_train
from prediff.utils.layout import layout_to_in_out_slice
from prediff.utils.path import (
    default_exps_dir,
    default_pretrained_vae_dir,
    default_pretrained_earthformerunet_dir,
    default_pretrained_alignment_dir, )
from prediff.taming import AutoencoderKL
from prediff.models.cuboid_transformer import CuboidTransformerUNet
from prediff.diffusion.latent_diffusion import LatentDiffusion
from prediff.diffusion.knowledge_alignment.sevir import SEVIRAvgIntensityAlignment


pytorch_state_dict_name = "sevirlr_earthformerunet.pt"


def get_alignment_kwargs_avg_x(context_seq=None, target_seq=None, ):
    r"""
    Please customize this function for generating knowledge "avg_x_gt"
    that guides the inference.
    E.g., this function uses 2.0 ground-truth future average intensity as "avg_x_gt" for demonstration.

    Parameters
    ----------
    context_seq:    torch.Tensor, aka "y"
    target_seq:     torch.Tensor, aka "x"

    Returns
    -------
    alignment_kwargs:   Dict
    """
    multiplier = 2.0
    batch_size = target_seq.shape[0]
    ret = torch.mean(target_seq.view(batch_size, -1),
                     dim=1, keepdim=True) * multiplier
    return {"avg_x_gt": ret}


class PreDiffSEVIRPLModule(LatentDiffusion):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc

        latent_model_cfg = OmegaConf.to_object(oc.model.latent_model)
        num_blocks = len(latent_model_cfg["depth"])
        if isinstance(latent_model_cfg["self_pattern"], str):
            block_attn_patterns = [latent_model_cfg["self_pattern"]] * num_blocks
        else:
            block_attn_patterns = OmegaConf.to_container(latent_model_cfg["self_pattern"])
        latent_model = CuboidTransformerUNet(
            input_shape=latent_model_cfg["input_shape"],
            target_shape=latent_model_cfg["target_shape"],
            base_units=latent_model_cfg["base_units"],
            scale_alpha=latent_model_cfg["scale_alpha"],
            num_heads=latent_model_cfg["num_heads"],
            attn_drop=latent_model_cfg["attn_drop"],
            proj_drop=latent_model_cfg["proj_drop"],
            ffn_drop=latent_model_cfg["ffn_drop"],
            # inter-attn downsample/upsample
            downsample=latent_model_cfg["downsample"],
            downsample_type=latent_model_cfg["downsample_type"],
            upsample_type=latent_model_cfg["upsample_type"],
            upsample_kernel_size=latent_model_cfg["upsample_kernel_size"],
            # attention
            depth=latent_model_cfg["depth"],
            block_attn_patterns=block_attn_patterns,
            # global vectors
            num_global_vectors=latent_model_cfg["num_global_vectors"],
            use_global_vector_ffn=latent_model_cfg["use_global_vector_ffn"],
            use_global_self_attn=latent_model_cfg["use_global_self_attn"],
            separate_global_qkv=latent_model_cfg["separate_global_qkv"],
            global_dim_ratio=latent_model_cfg["global_dim_ratio"],
            # misc
            ffn_activation=latent_model_cfg["ffn_activation"],
            gated_ffn=latent_model_cfg["gated_ffn"],
            norm_layer=latent_model_cfg["norm_layer"],
            padding_type=latent_model_cfg["padding_type"],
            checkpoint_level=latent_model_cfg["checkpoint_level"],
            pos_embed_type=latent_model_cfg["pos_embed_type"],
            use_relative_pos=latent_model_cfg["use_relative_pos"],
            self_attn_use_final_proj=latent_model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=latent_model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=latent_model_cfg["ffn_linear_init_mode"],
            ffn2_linear_init_mode=latent_model_cfg["ffn2_linear_init_mode"],
            attn_proj_linear_init_mode=latent_model_cfg["attn_proj_linear_init_mode"],
            conv_init_mode=latent_model_cfg["conv_init_mode"],
            down_linear_init_mode=latent_model_cfg["down_up_linear_init_mode"],
            up_linear_init_mode=latent_model_cfg["down_up_linear_init_mode"],
            global_proj_linear_init_mode=latent_model_cfg["global_proj_linear_init_mode"],
            norm_init_mode=latent_model_cfg["norm_init_mode"],
            # timestep embedding for diffusion
            time_embed_channels_mult=latent_model_cfg["time_embed_channels_mult"],
            time_embed_use_scale_shift_norm=latent_model_cfg["time_embed_use_scale_shift_norm"],
            time_embed_dropout=latent_model_cfg["time_embed_dropout"],
            unet_res_connect=latent_model_cfg["unet_res_connect"], )

        vae_cfg = OmegaConf.to_object(oc.model.vae)
        first_stage_model = AutoencoderKL(
            down_block_types=vae_cfg["down_block_types"],
            in_channels=vae_cfg["in_channels"],
            block_out_channels=vae_cfg["block_out_channels"],
            act_fn=vae_cfg["act_fn"],
            latent_channels=vae_cfg["latent_channels"],
            up_block_types=vae_cfg["up_block_types"],
            norm_num_groups=vae_cfg["norm_num_groups"],
            layers_per_block=vae_cfg["layers_per_block"],
            out_channels=vae_cfg["out_channels"], )
        pretrained_ckpt_path = vae_cfg["pretrained_ckpt_path"]
        if pretrained_ckpt_path is not None:
            state_dict = torch.load(os.path.join(default_pretrained_vae_dir, vae_cfg["pretrained_ckpt_path"]),
                                    map_location=torch.device("cpu"))
            first_stage_model.load_state_dict(state_dict=state_dict)
        else:
            warnings.warn(f"Pretrained weights for `AutoencoderKL` not set. Run for sanity check only.")

        diffusion_cfg = OmegaConf.to_object(oc.model.diffusion)
        super(PreDiffSEVIRPLModule, self).__init__(
            torch_nn_module=latent_model,
            layout=oc.layout.layout,
            data_shape=diffusion_cfg["data_shape"],
            timesteps=diffusion_cfg["timesteps"],
            beta_schedule=diffusion_cfg["beta_schedule"],
            loss_type=self.oc.optim.loss_type,
            monitor=self.oc.optim.monitor,
            use_ema=diffusion_cfg["use_ema"],
            log_every_t=diffusion_cfg["log_every_t"],
            clip_denoised=diffusion_cfg["clip_denoised"],
            linear_start=diffusion_cfg["linear_start"],
            linear_end=diffusion_cfg["linear_end"],
            cosine_s=diffusion_cfg["cosine_s"],
            given_betas=diffusion_cfg["given_betas"],
            original_elbo_weight=diffusion_cfg["original_elbo_weight"],
            v_posterior=diffusion_cfg["v_posterior"],
            l_simple_weight=diffusion_cfg["l_simple_weight"],
            parameterization=diffusion_cfg["parameterization"],
            learn_logvar=diffusion_cfg["learn_logvar"],
            logvar_init=diffusion_cfg["logvar_init"],
            # latent diffusion
            latent_shape=diffusion_cfg["latent_shape"],
            first_stage_model=first_stage_model,
            cond_stage_model=diffusion_cfg["cond_stage_model"],
            num_timesteps_cond=diffusion_cfg["num_timesteps_cond"],
            cond_stage_trainable=diffusion_cfg["cond_stage_trainable"],
            cond_stage_forward=diffusion_cfg["cond_stage_forward"],
            scale_by_std=diffusion_cfg["scale_by_std"],
            scale_factor=diffusion_cfg["scale_factor"], )
        # knowledge alignment
        knowledge_alignment_cfg = OmegaConf.to_object(oc.model.align)
        self.alignment_type = knowledge_alignment_cfg["alignment_type"]
        self.use_alignment = self.alignment_type is not None
        if self.use_alignment:
            alignment_ckpt_path = os.path.join(default_pretrained_alignment_dir, knowledge_alignment_cfg["model_ckpt_path"])
            self.alignment_obj = SEVIRAvgIntensityAlignment(
                alignment_type=knowledge_alignment_cfg["alignment_type"],
                guide_scale=knowledge_alignment_cfg["guide_scale"],
                model_type=knowledge_alignment_cfg["model_type"],
                model_args=knowledge_alignment_cfg["model_args"],
                model_ckpt_path=alignment_ckpt_path, )
            disable_train(self.alignment_obj.model)
            self.alignment_model = self.alignment_obj.model
            alignment_fn = self.alignment_obj.get_mean_shift
        else:
            alignment_fn = None
        self.set_alignment(alignment_fn=alignment_fn)
        # lr_scheduler
        self.total_num_steps = total_num_steps
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.eval.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.eval.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.eval.test_example_data_idx_list)
        self.eval_example_only = oc.eval.eval_example_only

        if self.oc.eval.eval_unaligned:
            self.valid_mse = torchmetrics.MeanSquaredError()
            self.valid_mae = torchmetrics.MeanAbsoluteError()
            self.valid_score = SEVIRSkillScore(
                mode=self.oc.dataset.metrics_mode,
                seq_len=self.oc.layout.out_len,
                layout=self.layout,
                threshold_list=self.oc.dataset.threshold_list,
                metrics_list=self.oc.dataset.metrics_list,
                eps=1e-4, )
            self.test_mse = torchmetrics.MeanSquaredError()
            self.test_mae = torchmetrics.MeanAbsoluteError()
            self.test_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
            self.test_score = SEVIRSkillScore(
                mode=self.oc.dataset.metrics_mode,
                seq_len=self.oc.layout.out_len,
                layout=self.layout,
                threshold_list=self.oc.dataset.threshold_list,
                metrics_list=self.oc.dataset.metrics_list,
                eps=1e-4, )
            self.test_fvd = FrechetVideoDistance(
                feature=self.oc.eval.fvd_features,
                layout=self.layout,
                reset_real_features=False,
                normalize=False,
                auto_t=True, )
        if self.oc.eval.eval_aligned:
            self.valid_aligned_mse = torchmetrics.MeanSquaredError()
            self.valid_aligned_mae = torchmetrics.MeanAbsoluteError()
            self.valid_aligned_score = SEVIRSkillScore(
                mode=self.oc.dataset.metrics_mode,
                seq_len=self.oc.layout.out_len,
                layout=self.layout,
                threshold_list=self.oc.dataset.threshold_list,
                metrics_list=self.oc.dataset.metrics_list,
                eps=1e-4, )
            self.test_aligned_mse = torchmetrics.MeanSquaredError()
            self.test_aligned_mae = torchmetrics.MeanAbsoluteError()
            self.test_aligned_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure()
            self.test_aligned_score = SEVIRSkillScore(
                mode=self.oc.dataset.metrics_mode,
                seq_len=self.oc.layout.out_len,
                layout=self.layout,
                threshold_list=self.oc.dataset.threshold_list,
                metrics_list=self.oc.dataset.metrics_list,
                eps=1e-4, )
            self.test_aligned_fvd = FrechetVideoDistance(
                feature=self.oc.eval.fvd_features,
                layout=self.layout,
                reset_real_features=False,
                normalize=False,
                auto_t=True, )

        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(default_exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)
        self.npy_save_dir = os.path.join(self.save_dir, "npy")
        os.makedirs(self.npy_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.eval = self.get_eval_config()
        oc.model = self.get_model_config()
        oc.dataset = self.get_dataset_config()
        if oc_from_file is not None:
            # oc = apply_omegaconf_overrides(oc, oc_from_file)
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_layout_config():
        cfg = OmegaConf.create()
        cfg.in_len = 10
        cfg.out_len = 20
        cfg.img_height = 128
        cfg.img_width = 128
        cfg.data_channels = 4
        cfg.layout = "NTHWC"
        return cfg

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        layout_cfg = cls.get_layout_config()

        cfg.diffusion = OmegaConf.create()
        cfg.diffusion.data_shape = (layout_cfg.out_len,
                                    layout_cfg.img_height,
                                    layout_cfg.img_width,
                                    layout_cfg.data_channels)
        cfg.diffusion.timesteps = 1000
        cfg.diffusion.beta_schedule = "linear"
        cfg.diffusion.use_ema = True
        cfg.diffusion.log_every_t = 100  # log every `log_every_t` timesteps. Must be smaller than `timesteps`.
        cfg.diffusion.clip_denoised = False
        cfg.diffusion.linear_start = 1e-4
        cfg.diffusion.linear_end = 2e-2
        cfg.diffusion.cosine_s = 8e-3
        cfg.diffusion.given_betas = None
        cfg.diffusion.original_elbo_weight = 0.
        cfg.diffusion.v_posterior = 0.
        cfg.diffusion.l_simple_weight = 1.
        cfg.diffusion.parameterization = "eps"
        cfg.diffusion.learn_logvar = None
        cfg.diffusion.logvar_init = 0.
        # latent diffusion
        cfg.diffusion.latent_shape = [10, 16, 16, 4]
        cfg.diffusion.cond_stage_model = "__is_first_stage__"
        cfg.diffusion.num_timesteps_cond = None
        cfg.diffusion.cond_stage_trainable = False
        cfg.diffusion.cond_stage_forward = None
        cfg.diffusion.scale_by_std = False
        cfg.diffusion.scale_factor = 1.0
        cfg.diffusion.latent_cond_shape = [10, 16, 16, 4]
        # knowledge alignment
        cfg.align = OmegaConf.create()
        cfg.align.alignment_type = None
        cfg.align.guide_scale = 1.0
        cfg.align.model_type = "cuboid"
        cfg.align.model_ckpt_path = "tmp.pt"
        cfg.align.model_args = OmegaConf.create()
        # Earthformer
        cfg.align.model_args.input_shape = [6, 16, 16, 4]
        cfg.align.model_args.out_channels = 2
        cfg.align.model_args.base_units = 16
        cfg.align.model_args.block_units = None
        cfg.align.model_args.scale_alpha = 1.0
        cfg.align.model_args.depth = [1, 1]
        cfg.align.model_args.downsample = 2
        cfg.align.model_args.downsample_type = "patch_merge"
        cfg.align.model_args.block_attn_patterns = "axial"
        cfg.align.model_args.num_heads = 4
        cfg.align.model_args.attn_drop = 0.0
        cfg.align.model_args.proj_drop = 0.0
        cfg.align.model_args.ffn_drop = 0.0
        cfg.align.model_args.ffn_activation = "gelu"
        cfg.align.model_args.gated_ffn = False
        cfg.align.model_args.norm_layer = "layer_norm"
        cfg.align.model_args.use_inter_ffn = True
        cfg.align.model_args.hierarchical_pos_embed = False
        cfg.align.model_args.pos_embed_type = 't+h+w'
        cfg.align.model_args.padding_type = "zero"
        cfg.align.model_args.checkpoint_level = 0
        cfg.align.model_args.use_relative_pos = True
        cfg.align.model_args.self_attn_use_final_proj = True
        # global vectors
        cfg.align.model_args.num_global_vectors = 0
        cfg.align.model_args.use_global_vector_ffn = True
        cfg.align.model_args.use_global_self_attn = False
        cfg.align.model_args.separate_global_qkv = False
        cfg.align.model_args.global_dim_ratio = 1
        # initialization
        cfg.align.model_args.attn_linear_init_mode = "0"
        cfg.align.model_args.ffn_linear_init_mode = "0"
        cfg.align.model_args.ffn2_linear_init_mode = "2"
        cfg.align.model_args.attn_proj_linear_init_mode = "2"
        cfg.align.model_args.conv_init_mode = "0"
        cfg.align.model_args.down_linear_init_mode = "0"
        cfg.align.model_args.global_proj_linear_init_mode = "2"
        cfg.align.model_args.norm_init_mode = "0"
        # timestep embedding for diffusion
        cfg.align.model_args.time_embed_channels_mult = 4
        cfg.align.model_args.time_embed_use_scale_shift_norm = False
        cfg.align.model_args.time_embed_dropout = 0.0
        # readout
        cfg.align.model_args.pool = "attention"
        cfg.align.model_args.readout_seq = True
        cfg.align.model_args.out_len = 6

        cfg.latent_model = OmegaConf.create()
        cfg.latent_model.input_shape = [10, 16, 16, 4]
        cfg.latent_model.target_shape = [10, 16, 16, 4]
        cfg.latent_model.base_units = 4
        # block_units = null
        cfg.latent_model.scale_alpha = 1.0
        cfg.latent_model.num_heads = 4
        cfg.latent_model.attn_drop = 0.1
        cfg.latent_model.proj_drop = 0.1
        cfg.latent_model.ffn_drop = 0.1
        # inter-attn downsample/upsample
        cfg.latent_model.downsample = 2
        cfg.latent_model.downsample_type = "patch_merge"
        cfg.latent_model.upsample_type = "upsample"
        cfg.latent_model.upsample_kernel_size = 3
        # cuboid attention
        cfg.latent_model.depth = [1, 1]
        cfg.latent_model.self_pattern = "axial"
        # global vectors
        cfg.latent_model.num_global_vectors = 0
        cfg.latent_model.use_dec_self_global = False
        cfg.latent_model.dec_self_update_global = True
        cfg.latent_model.use_dec_cross_global = False
        cfg.latent_model.use_global_vector_ffn = False
        cfg.latent_model.use_global_self_attn = True
        cfg.latent_model.separate_global_qkv = True
        cfg.latent_model.global_dim_ratio = 1
        # mise
        cfg.latent_model.ffn_activation = "gelu"
        cfg.latent_model.gated_ffn = False
        cfg.latent_model.norm_layer = "layer_norm"
        cfg.latent_model.padding_type = "zeros"
        cfg.latent_model.pos_embed_type = "t+h+w"
        cfg.latent_model.checkpoint_level = 0
        cfg.latent_model.use_relative_pos = True
        cfg.latent_model.self_attn_use_final_proj = True
        # initialization
        cfg.latent_model.attn_linear_init_mode = "0"
        cfg.latent_model.ffn_linear_init_mode = "0"
        cfg.latent_model.ffn2_linear_init_mode = "2"
        cfg.latent_model.attn_proj_linear_init_mode = "2"
        cfg.latent_model.conv_init_mode = "0"
        cfg.latent_model.down_up_linear_init_mode = "0"
        cfg.latent_model.global_proj_linear_init_mode = "2"
        cfg.latent_model.norm_init_mode = "0"
        # timestep embedding for diffusion
        cfg.latent_model.time_embed_channels_mult = 4
        cfg.latent_model.time_embed_use_scale_shift_norm = False
        cfg.latent_model.time_embed_dropout = 0.0
        cfg.latent_model.unet_res_connect = True

        cfg.vae = OmegaConf.create()
        cfg.vae.data_channels = layout_cfg.data_channels
        # from stable-diffusion-v1-5
        cfg.vae.down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D']
        cfg.vae.in_channels = cfg.vae.data_channels
        cfg.vae.block_out_channels = [128, 256, 512, 512]
        cfg.vae.act_fn = 'silu'
        cfg.vae.latent_channels = 4
        cfg.vae.up_block_types = ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D']
        cfg.vae.norm_num_groups = 32
        cfg.vae.layers_per_block = 2
        cfg.vae.out_channels = cfg.vae.data_channels
        return cfg

    @classmethod
    def get_dataset_config(cls):
        cfg = OmegaConf.create()
        cfg.dataset_name = "sevir_lr"
        cfg.img_height = 128
        cfg.img_width = 128
        cfg.in_len = 7
        cfg.out_len = 6
        cfg.seq_len = 13
        cfg.plot_stride = 1
        cfg.interval_real_time = 10
        cfg.sample_mode = "sequent"
        cfg.stride = cfg.out_len
        cfg.layout = "NTHWC"
        cfg.start_date = None
        cfg.train_val_split_date = (2019, 1, 1)
        cfg.train_test_split_date = (2019, 6, 1)
        cfg.end_date = None
        cfg.metrics_mode = "0"
        cfg.metrics_list = ('csi', 'pod', 'sucr', 'bias')
        cfg.threshold_list = (16, 74, 133, 160, 181, 219)
        cfg.aug_mode = "1"
        return cfg

    @staticmethod
    def get_optim_config():
        cfg = OmegaConf.create()
        cfg.seed = None
        cfg.total_batch_size = 32
        cfg.micro_batch_size = 8
        cfg.float32_matmul_precision = "high"

        cfg.method = "adamw"
        cfg.lr = 1.0E-6
        cfg.wd = 1.0E-2
        cfg.betas = (0.9, 0.999)
        cfg.gradient_clip_val = 1.0
        cfg.max_epochs = 50
        cfg.loss_type = "l2"
        # scheduler
        cfg.warmup_percentage = 0.2
        cfg.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        cfg.min_lr_ratio = 1.0E-3
        cfg.warmup_min_lr_ratio = 0.0
        # early stopping
        cfg.monitor = "valid_loss_epoch"
        cfg.early_stop = False
        cfg.early_stop_mode = "min"
        cfg.early_stop_patience = 5
        cfg.save_top_k = 1
        return cfg

    @staticmethod
    def get_logging_config():
        cfg = OmegaConf.create()
        cfg.logging_prefix = "PreDiff"
        cfg.monitor_lr = True
        cfg.monitor_device = False
        cfg.track_grad_norm = -1
        cfg.use_wandb = False
        cfg.profiler = None
        cfg.save_npy = False
        return cfg

    @staticmethod
    def get_trainer_config():
        cfg = OmegaConf.create()
        cfg.check_val_every_n_epoch = 1
        cfg.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        cfg.precision = 32
        cfg.find_unused_parameters = True
        cfg.num_sanity_val_steps = 2
        return cfg

    @staticmethod
    def get_eval_config():
        cfg = OmegaConf.create()
        cfg.train_example_data_idx_list = [0, ]
        cfg.val_example_data_idx_list = [0, ]
        cfg.test_example_data_idx_list = [0, ]
        cfg.eval_example_only = False
        cfg.eval_aligned = True
        cfg.eval_unaligned = True
        cfg.num_samples_per_context = 1
        cfg.font_size = 20
        cfg.label_offset = (-0.5, 0.5)
        cfg.label_avg_int = False
        cfg.fvd_features = 400
        return cfg

    def configure_optimizers(self):
        optim_cfg = self.oc.optim
        params = list(self.torch_nn_module.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)

        if optim_cfg.method == "adamw":
            optimizer = torch.optim.AdamW(params, lr=optim_cfg.lr, betas=optim_cfg.betas)
        else:
            raise NotImplementedError(f"opimization method {optim_cfg.method} not supported.")

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))
        if optim_cfg.lr_scheduler_mode == 'none':
            return {'optimizer': optimizer}
        else:
            if optim_cfg.lr_scheduler_mode == 'cosine':
                warmup_scheduler = LambdaLR(optimizer,
                                            lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                    min_lr_ratio=optim_cfg.warmup_min_lr_ratio))
                cosine_scheduler = CosineAnnealingLR(optimizer,
                                                     T_max=(self.total_num_steps - warmup_iter),
                                                     eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr)
                lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                            milestones=[warmup_iter])
                lr_scheduler_config = {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            else:
                raise NotImplementedError
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        if self.oc.logging.profiler is None:
            profiler = None
        elif self.oc.logging.profiler == "pytorch":
            profiler = PyTorchProfiler(filename=f"{self.oc.logging.logging_prefix}_PyTorchProfiler.log")
        else:
            raise NotImplementedError
        checkpoint_callback = ModelCheckpoint(
            monitor=self.oc.optim.monitor,
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="{epoch:03d}",
            auto_insert_metric_name=False,
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            profiler=profiler,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            strategy=DDPStrategy(find_unused_parameters=self.oc.trainer.find_unused_parameters),
            # strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
            # misc
            num_sanity_val_steps=self.oc.trainer.num_sanity_val_steps,
            inference_mode=False,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        epoch: int
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_sevir_datamodule(dataset_cfg,
                             micro_batch_size: int = 1,
                             num_workers: int = 8):
        dm = SEVIRLightningDataModule(
            seq_len=dataset_cfg["seq_len"],
            sample_mode=dataset_cfg["sample_mode"],
            stride=dataset_cfg["stride"],
            batch_size=micro_batch_size,
            layout=dataset_cfg["layout"],
            output_type=np.float32,
            preprocess=True,
            rescale_method="01",
            verbose=False,
            aug_mode=dataset_cfg["aug_mode"],
            ret_contiguous=False,
            # datamodule_only
            dataset_name=dataset_cfg["dataset_name"],
            start_date=dataset_cfg["start_date"],
            train_test_split_date=dataset_cfg["train_test_split_date"],
            end_date=dataset_cfg["end_date"],
            val_ratio=dataset_cfg["val_ratio"],
            num_workers=num_workers, )
        return dm

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.oc.layout.layout,
                in_len=self.oc.layout.in_len,
                out_len=self.oc.layout.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.oc.layout.layout,
                in_len=self.oc.layout.in_len,
                out_len=self.oc.layout.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

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
        return self._get_input_sevirlr(batch=batch, return_verbose=kwargs.get("return_verbose", False))

    @torch.no_grad()
    def _get_input_sevirlr(self, batch, return_verbose=False):
        seq = batch
        in_seq = seq[self.in_slice]
        out_seq = seq[self.out_slice].contiguous()
        if return_verbose:
            return out_seq, {"y": in_seq}, in_seq
        else:
            return out_seq, {"y": in_seq}

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if self.current_epoch % self.oc.trainer.check_val_every_n_epoch == 0 \
                and self.local_rank == 0:
            if data_idx in self.train_example_data_idx_list:
                target_seq, cond, context_seq = \
                    self.get_input(batch, return_verbose=True)
                aligned_pred_seq_list = []
                aligned_pred_label_list = []
                pred_seq_list = []
                pred_label_list = []
                for i in range(self.oc.eval.num_samples_per_context):
                    # aligned sampling
                    if self.use_alignment and self.oc.eval.eval_aligned:
                        if self.alignment_type == "avg_x":
                            alignment_kwargs = get_alignment_kwargs_avg_x(context_seq=context_seq,
                                                                          target_seq=target_seq)
                        else:
                            raise NotImplementedError
                        pred_seq = self.sample(
                            cond=cond,
                            batch_size=micro_batch_size,
                            return_intermediates=False,
                            use_alignment=True,
                            alignment_kwargs=alignment_kwargs,
                            verbose=False, ).contiguous()
                        aligned_pred_seq_list.append(pred_seq[0].detach().float().cpu().numpy())
                        aligned_pred_label_list.append(f"{self.oc.logging.logging_prefix}_aligned_pred_{i}")
                    # no alignment
                    if self.oc.eval.eval_unaligned:
                        pred_seq = self.sample(
                            cond=cond,
                            batch_size=micro_batch_size,
                            return_intermediates=False,
                            verbose=False, ).contiguous()
                        pred_seq_list.append(pred_seq[0].detach().float().cpu().numpy())
                        pred_label_list.append(f"{self.oc.logging.logging_prefix}_pred_{i}")
                pred_seq_list = aligned_pred_seq_list + pred_seq_list
                pred_label_list = aligned_pred_label_list + pred_label_list
                self.save_vis_step_end(
                    data_idx=data_idx,
                    context_seq=context_seq[0].detach().float().cpu().numpy(),
                    target_seq=target_seq[0].detach().float().cpu().numpy(),
                    pred_seq=pred_seq_list,
                    pred_label=pred_label_list,
                    mode="train", )
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self(batch)
        with self.ema_scope():
            _, loss_dict_ema = self(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            target_seq, cond, context_seq = \
                self.get_input(batch, return_verbose=True)
            aligned_pred_seq_list = []
            aligned_pred_label_list = []
            pred_seq_list = []
            pred_label_list = []
            for i in range(self.oc.eval.num_samples_per_context):
                # aligned sampling
                if self.use_alignment and self.oc.eval.eval_aligned:
                    if self.alignment_type == "avg_x":
                        alignment_kwargs = get_alignment_kwargs_avg_x(context_seq=context_seq,
                                                                      target_seq=target_seq)
                    else:
                        raise NotImplementedError
                    pred_seq = self.sample(
                        cond=cond,
                        batch_size=micro_batch_size,
                        return_intermediates=False,
                        use_alignment=True,
                        alignment_kwargs=alignment_kwargs,
                        verbose=False, ).contiguous()
                    aligned_pred_seq_list.append(pred_seq[0].detach().float().cpu().numpy())
                    aligned_pred_label_list.append(f"{self.oc.logging.logging_prefix}_aligned_pred_{i}")
                    if pred_seq.dtype is not torch.float:
                        pred_seq = pred_seq.float()
                    self.valid_aligned_mse(pred_seq, target_seq)
                    self.valid_aligned_mae(pred_seq, target_seq)
                    self.valid_aligned_score.update(pred_seq, target_seq)
                # no alignment
                if self.oc.eval.eval_unaligned:
                    pred_seq = self.sample(
                        cond=cond,
                        batch_size=micro_batch_size,
                        return_intermediates=False,
                        verbose=False, ).contiguous()
                    pred_seq_list.append(pred_seq[0].detach().float().cpu().numpy())
                    pred_label_list.append(f"{self.oc.logging.logging_prefix}_pred_{i}")
                    if pred_seq.dtype is not torch.float:
                        pred_seq = pred_seq.float()
                    self.valid_mse(pred_seq, target_seq)
                    self.valid_mae(pred_seq, target_seq)
                    self.valid_score.update(pred_seq, target_seq)
            pred_seq_list = aligned_pred_seq_list + pred_seq_list
            pred_label_list = aligned_pred_label_list + pred_label_list
            self.save_vis_step_end(
                data_idx=data_idx,
                context_seq=context_seq[0].detach().float().cpu().numpy(),
                target_seq=target_seq[0].detach().float().cpu().numpy(),
                pred_seq=pred_seq_list,
                pred_label=pred_label_list,
                mode="val",
                suffix=f"_rank{self.local_rank}", )

    def on_validation_epoch_end(self):
        if self.oc.eval.eval_unaligned:
            valid_mse = self.valid_mse.compute()
            valid_mae = self.valid_mae.compute()
            valid_score = self.valid_score.compute()
            valid_loss = -valid_score["avg"]["csi"]

            self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_score_epoch_end(score_dict=valid_score, prefix="valid")
            self.valid_mse.reset()
            self.valid_mae.reset()
            self.valid_score.reset()
        if self.oc.eval.eval_aligned:
            valid_mse = self.valid_aligned_mse.compute()
            valid_mae = self.valid_aligned_mae.compute()
            valid_score = self.valid_aligned_score.compute()
            valid_loss = -valid_score["avg"]["csi"]

            self.log('valid_aligned_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True,
                     sync_dist=True)
            self.log('valid_aligned_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('valid_aligned_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_score_epoch_end(score_dict=valid_score, prefix="valid_aligned")
            self.valid_aligned_mse.reset()
            self.valid_aligned_mae.reset()
            self.valid_aligned_score.reset()

    def test_step(self, batch, batch_idx):
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            target_seq, cond, context_seq = \
                self.get_input(batch, return_verbose=True)
            target_seq_bchw = rearrange(target_seq, "b t h w c -> (b t) c h w")
            aligned_pred_seq_list = []
            aligned_pred_label_list = []
            pred_seq_list = []
            pred_label_list = []
            for i in range(self.oc.eval.num_samples_per_context):
                # aligned sampling
                if self.use_alignment and self.oc.eval.eval_aligned:
                    if self.alignment_type == "avg_x":
                        alignment_kwargs = get_alignment_kwargs_avg_x(context_seq=context_seq,
                                                                      target_seq=target_seq)
                    else:
                        raise NotImplementedError
                    pred_seq = self.sample(
                        cond=cond,
                        batch_size=micro_batch_size,
                        return_intermediates=False,
                        use_alignment=True,
                        alignment_kwargs=alignment_kwargs,
                        verbose=False, ).contiguous()
                    if self.oc.logging.save_npy:
                        npy_path = os.path.join(self.npy_save_dir,
                                                f"batch{batch_idx}_rank{self.local_rank}_sample{i}_aligned.npy")
                        np.save(npy_path, pred_seq.detach().float().cpu().numpy())
                    aligned_pred_seq_list.append(pred_seq[0].detach().float().cpu().numpy())
                    aligned_pred_label_list.append(f"{self.oc.logging.logging_prefix}_aligned_pred_{i}")
                    if pred_seq.dtype is not torch.float:
                        pred_seq = pred_seq.float()
                    self.test_aligned_mse(pred_seq, target_seq)
                    self.test_aligned_mae(pred_seq, target_seq)
                    self.test_aligned_score.update(pred_seq, target_seq)
                    self.test_aligned_fvd.update(pred_seq, real=False)
                    pred_seq_bchw = rearrange(pred_seq, "b t h w c -> (b t) c h w")
                    self.test_aligned_ssim(pred_seq_bchw, target_seq_bchw)
                # no alignment
                if self.oc.eval.eval_unaligned:
                    pred_seq = self.sample(
                        cond=cond,
                        batch_size=micro_batch_size,
                        return_intermediates=False,
                        verbose=False, ).contiguous()
                    if self.oc.logging.save_npy:
                        npy_path = os.path.join(self.npy_save_dir,
                                                f"batch{batch_idx}_rank{self.local_rank}_sample{i}.npy")
                        np.save(npy_path, pred_seq.detach().float().cpu().numpy())
                    pred_seq_list.append(pred_seq[0].detach().float().cpu().numpy())
                    pred_label_list.append(f"{self.oc.logging.logging_prefix}_pred_{i}")
                    if pred_seq.dtype is not torch.float:
                        pred_seq = pred_seq.float()
                    self.test_mse(pred_seq, target_seq)
                    self.test_mae(pred_seq, target_seq)
                    self.test_score.update(pred_seq, target_seq)
                    self.test_fvd.update(pred_seq, real=False)
                    pred_seq_bchw = rearrange(pred_seq, "b t h w c -> (b t) c h w")
                    self.test_ssim(pred_seq_bchw, target_seq_bchw)
            if self.use_alignment and self.oc.eval.eval_aligned:
                self.test_aligned_fvd.update(target_seq, real=True)
            if self.oc.eval.eval_unaligned:
                self.test_fvd.update(target_seq, real=True)
            pred_seq_list = aligned_pred_seq_list + pred_seq_list
            pred_label_list = aligned_pred_label_list + pred_label_list
            self.save_vis_step_end(
                data_idx=data_idx,
                context_seq=context_seq[0].detach().float().cpu().numpy(),
                target_seq=target_seq[0].detach().float().cpu().numpy(),
                pred_seq=pred_seq_list,
                pred_label=pred_label_list,
                mode="test",
                suffix=f"_rank{self.local_rank}", )

    def on_test_epoch_end(self):
        if self.oc.eval.eval_unaligned:
            test_mse = self.test_mse.compute()
            test_mae = self.test_mae.compute()
            test_ssim = self.test_ssim.compute()
            test_score = self.test_score.compute()
            test_fvd = self.test_fvd.compute()

            self.log('test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test_ssim_epoch', test_ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_score_epoch_end(score_dict=test_score, prefix="test")
            self.log('test_fvd_epoch', test_fvd, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.test_mse.reset()
            self.test_mae.reset()
            self.test_ssim.reset()
            self.test_score.reset()
            self.test_fvd.reset()
        if self.oc.eval.eval_aligned:
            test_mse = self.test_aligned_mse.compute()
            test_mae = self.test_aligned_mae.compute()
            test_ssim = self.test_aligned_ssim.compute()
            test_score = self.test_aligned_score.compute()
            test_fvd = self.test_aligned_fvd.compute()

            self.log('test_aligned_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test_aligned_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test_aligned_ssim_epoch', test_ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_score_epoch_end(score_dict=test_score, prefix="test_aligned")
            self.log('test_aligned_fvd_epoch', test_fvd, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.test_aligned_mse.reset()
            self.test_aligned_mae.reset()
            self.test_aligned_ssim.reset()
            self.test_aligned_score.reset()
            self.test_aligned_fvd.reset()

    def save_vis_step_end(
            self,
            data_idx: int,
            context_seq: np.ndarray,
            target_seq: np.ndarray,
            pred_seq: Union[np.ndarray, Sequence[np.ndarray]],
            pred_label: Union[str, Sequence[str]] = None,
            label_mode: str = "name",
            mode: str = "train",
            prefix: str = "",
            suffix: str = "", ):
        r"""
        Parameters
        ----------
        data_idx
        context_seq, target_seq, pred_seq:   np.ndarray
            layout should not include batch
        mode:   str
        """
        if mode == "train":
            example_data_idx_list = self.train_example_data_idx_list
        elif mode == "val":
            example_data_idx_list = self.val_example_data_idx_list
        elif mode == "test":
            example_data_idx_list = self.test_example_data_idx_list
        else:
            raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
        if label_mode == "name":
            # use the given label
            context_label = "context"
            target_label = "target"
        elif label_mode == "avg_int":
            context_label = f"context\navg_int={np.mean(context_seq):.4f}"
            target_label = f"target\navg_int={np.mean(target_seq):.4f}"
            if isinstance(pred_label, Sequence):
                pred_label = [f"{label}\navg_int={np.mean(seq):.4f}" for label, seq in zip(pred_label, pred_seq)]
            elif isinstance(pred_label, str):
                pred_label = f"{pred_label}\navg_int={np.mean(pred_seq):.4f}"
            else:
                raise TypeError(f"Wrong pred_label type {type(pred_label)}! must be in [str, Sequence[str]].")
        else:
            raise NotImplementedError
        if isinstance(pred_seq, Sequence):
            seq_list = [context_seq, target_seq] + list(pred_seq)
            label_list = [context_label, target_label] + pred_label
        else:
            seq_list = [context_seq, target_seq, pred_seq]
            label_list = [context_label, target_label, pred_label]
        if data_idx in example_data_idx_list:
            png_save_name = f"{prefix}{mode}_epoch_{self.current_epoch}_data_{data_idx}{suffix}.png"
            vis_sevir_seq(
                save_path=os.path.join(self.example_save_dir, png_save_name),
                seq=seq_list,
                label=label_list,
                interval_real_time=10,
                plot_stride=1, fs=self.oc.eval.fs,
                label_offset=self.oc.eval.label_offset,
                label_avg_int=self.oc.eval.label_avg_int, )

    def log_score_epoch_end(self, score_dict: Dict, prefix: str = "valid"):
        for metrics in self.oc.dataset.metrics_list:
            for thresh in self.oc.dataset.threshold_list:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                self.log(f"{prefix}_{metrics}_{thresh}_epoch", score_mean,
                         prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                self.log(f"{prefix}_{metrics}_avg_epoch", score_avg_mean,
                         prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # reference: https://lightning.ai/docs/pytorch/2.0.9/debug/debugging_intermediate.html#look-out-for-exploding-gradients
        if self.oc.logging.track_grad_norm != -1:
            norms = grad_norm(self.torch_nn_module, norm_type=self.oc.logging.track_grad_norm)
            self.log_dict(norms)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_sevirlr', type=str)
    parser.add_argument('--nodes', default=1, type=int,
                        help="Number of nodes in DDP training.")
    parser.add_argument('--gpus', default=1, type=int,
                        help="Number of GPUS per node in DDP training.")
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on SEVIR-LR.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.pretrained:
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "prediff_sevirlr_v1.yaml"))
        # Download pretrained weights
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_vae_name,
                                    save_dir=default_pretrained_vae_dir,
                                    exist_ok=False)
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_earthformerunet_name,
                                    save_dir=default_pretrained_earthformerunet_dir,
                                    exist_ok=False)
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_alignment_name,
                                    save_dir=default_pretrained_alignment_dir,
                                    exist_ok=False)
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
        float32_matmul_precision = oc_from_file.optim.float32_matmul_precision
    else:
        dataset_cfg = OmegaConf.to_object(PreDiffSEVIRPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.nodes * args.gpus)
        max_epochs = None
        seed = 0
        float32_matmul_precision = "high"
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed_everything(seed, workers=True)
    dm = PreDiffSEVIRPLModule.get_sevir_datamodule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=8, )
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.nodes * args.gpus)
    total_num_steps = PreDiffSEVIRPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = PreDiffSEVIRPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        num_nodes=args.nodes,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        # load Earthformer-UNet
        earthformerunet_ckpt_path = os.path.join(default_pretrained_earthformerunet_dir,
                                                 pretrained_sevirlr_earthformerunet_name)
        state_dict = torch.load(earthformerunet_ckpt_path,
                                map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)
        trainer.test(model=pl_module,
                     datamodule=dm)
    elif args.test:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            pl_ckpt = pl_load(path_or_url=ckpt_path,
                              map_location=torch.device("cpu"))
            # pl_state_dict = pl_ckpt["state_dict"]  # pl 1.x
            pl_state_dict = pl_ckpt
            model_kay = "torch_nn_module."
            model_state_dict = OrderedDict()
            for key, val in pl_state_dict.items():
                if key.startswith(model_kay):
                    model_state_dict[key.replace(model_kay, "")] = val
            pl_module.torch_nn_module.load_state_dict(model_state_dict)
        trainer.test(model=pl_module,
                     datamodule=dm, )
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(model=pl_module,
                    datamodule=dm,
                    ckpt_path=ckpt_path)
        # save state_dict of the latent diffusion model
        pl_ckpt = pl_load(path_or_url=trainer.checkpoint_callback.best_model_path,
                          map_location=torch.device("cpu"))
        # pl_state_dict = pl_ckpt["state_dict"]  # pl 1.x
        pl_state_dict = pl_ckpt
        model_kay = "torch_nn_module."
        state_dict = OrderedDict()
        unexpected_dict = OrderedDict()
        for key, val in pl_state_dict.items():
            if key.startswith(model_kay):
                state_dict[key.replace(model_kay, "")] = val
            else:
                unexpected_dict[key] = val
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        # test
        trainer.test(ckpt_path="best",
                     datamodule=dm)


if __name__ == "__main__":
    main()
