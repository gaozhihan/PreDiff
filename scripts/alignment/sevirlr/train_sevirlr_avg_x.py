import warnings
from typing import Sequence, Union, Dict
from shutil import copyfile
import inspect
from collections import OrderedDict
import multiprocessing
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

from prediff.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
from prediff.utils.pl_checkpoint import pl_load
from prediff.utils.download import (
    download_pretrained_weights,
    pretrained_sevirlr_vae_name,
    pretrained_sevirlr_earthformerunet_name,
    pretrained_sevirlr_alignment_name)
from prediff.utils.optim import warmup_lambda
from prediff.utils.layout import layout_to_in_out_slice
from prediff.utils.path import (
    default_exps_dir,
    default_pretrained_vae_dir,
    default_pretrained_earthformerunet_dir,
    default_pretrained_alignment_dir, )
from prediff.taming import AutoencoderKL
from prediff.diffusion.knowledge_alignment.alignment_pl import AlignmentPL
from prediff.diffusion.knowledge_alignment.sevir import SEVIRAvgIntensityAlignment


pytorch_state_dict_name = "sevirlr_alignment_avgx.pt"


class SEVIRAlignmentPLModule(AlignmentPL):

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

        knowledge_alignment_cfg = OmegaConf.to_object(oc.model.align)
        self.alignment_obj = SEVIRAvgIntensityAlignment(
            alignment_type=knowledge_alignment_cfg["alignment_type"],
            model_type=knowledge_alignment_cfg["model_type"],
            model_args=knowledge_alignment_cfg["model_args"],)

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
        super(SEVIRAlignmentPLModule, self).__init__(
            torch_nn_module=self.alignment_obj.model,
            target_fn=self.alignment_obj.model_objective,
            layout=oc.layout.layout,
            timesteps=diffusion_cfg["timesteps"],
            beta_schedule=diffusion_cfg["beta_schedule"],
            loss_type=self.oc.optim.loss_type,
            monitor=self.oc.optim.monitor,
            linear_start=diffusion_cfg["linear_start"],
            linear_end=diffusion_cfg["linear_end"],
            cosine_s=diffusion_cfg["cosine_s"],
            given_betas=diffusion_cfg["given_betas"],
            # latent diffusion
            first_stage_model=first_stage_model,
            cond_stage_model=diffusion_cfg["cond_stage_model"],
            num_timesteps_cond=diffusion_cfg["num_timesteps_cond"],
            cond_stage_trainable=diffusion_cfg["cond_stage_trainable"],
            cond_stage_forward=diffusion_cfg["cond_stage_forward"],
            scale_by_std=diffusion_cfg["scale_by_std"],
            scale_factor=diffusion_cfg["scale_factor"],)
        # lr_scheduler
        self.total_num_steps = total_num_steps
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

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
        cfg.diffusion.timesteps = 1000
        cfg.diffusion.beta_schedule = "linear"
        cfg.diffusion.linear_start = 1e-4
        cfg.diffusion.linear_end = 2e-2
        cfg.diffusion.cosine_s = 8e-3
        cfg.diffusion.given_betas = None
        # latent diffusion
        cfg.diffusion.cond_stage_model = "__is_first_stage__"
        cfg.diffusion.num_timesteps_cond = None
        cfg.diffusion.cond_stage_trainable = False
        cfg.diffusion.cond_stage_forward = None
        cfg.diffusion.scale_by_std = False
        cfg.diffusion.scale_factor = 1.0

        cfg.align = OmegaConf.create()
        cfg.align.alignment_type = "avg_x"
        cfg.align.model_type = "cuboid"
        cfg.align.model_args = OmegaConf.create()
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
        cfg.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine'
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
        cfg.logging_prefix = "SEVIR-LR_AvgX"
        cfg.monitor_lr = True
        cfg.monitor_device = False
        cfg.track_grad_norm = -1
        cfg.use_wandb = False
        cfg.profiler = None
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
        cfg.train_example_data_idx_list = []
        cfg.val_example_data_idx_list = []
        cfg.test_example_data_idx_list = []
        cfg.eval_example_only = False
        cfg.num_samples_per_context = 1
        cfg.save_gif = False
        cfg.gif_fps = 2.0
        return cfg

    def configure_optimizers(self):
        optim_cfg = self.oc.optim
        params = list(self.torch_nn_module.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())

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

    @property
    def intensity_avg_dims(self):
        if not hasattr(self, "_intensity_avg_dims"):
            self._intensity_avg_dims = tuple(self.oc.layout.layout.find(dim) for dim in "HWC")
        return self._intensity_avg_dims

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
        out_seq = seq[self.out_slice]
        if return_verbose:
            return out_seq, {"y": in_seq}, \
                   {"avg_x_gt": torch.mean(out_seq, dim=self.intensity_avg_dims)}
        else:
            return out_seq, {"y": in_seq}, {}

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # reference: https://lightning.ai/docs/pytorch/2.0.9/debug/debugging_intermediate.html#look-out-for-exploding-gradients
        if self.oc.logging.track_grad_norm != -1:
            norms = grad_norm(self.torch_nn_module, norm_type=self.oc.logging.track_grad_norm)
            self.log_dict(norms)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_sevirlr_avg_x', type=str)
    parser.add_argument('--nodes', default=1, type=int,
                        help="Number of nodes in DDP training.")
    parser.add_argument('--gpus', default=1, type=int,
                        help="Number of GPUS per node in DDP training.")
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on SEVIR-LR.')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
        float32_matmul_precision = oc_from_file.optim.float32_matmul_precision
    else:
        dataset_cfg = OmegaConf.to_object(SEVIRAlignmentPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.nodes * args.gpus)
        max_epochs = None
        seed = 0
        float32_matmul_precision = "high"
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed_everything(seed, workers=True)
    dm = SEVIRAlignmentPLModule.get_sevir_datamodule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=8, )
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.nodes * args.gpus)
    total_num_steps = SEVIRAlignmentPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = SEVIRAlignmentPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        num_nodes=args.nodes,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)
    if args.test:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        else:
            ckpt_path = None
        trainer.test(model=pl_module,
                     datamodule=dm,
                     ckpt_path=ckpt_path)
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
        # save state_dict of the latent diffusion model, i.e., EarthformerDiffusion
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
