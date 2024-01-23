import warnings
from shutil import copyfile
import inspect
from collections import OrderedDict
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
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
from prediff.taming import AutoencoderKL, LPIPSWithDiscriminator
from prediff.utils.optim import warmup_lambda
from prediff.utils.pl_checkpoint import pl_load
from prediff.utils.download import (
    download_pretrained_weights,
    pretrained_sevirlr_vae_name)
from prediff.utils.path import default_pretrained_vae_dir
from prediff.utils.path import default_exps_dir


pytorch_state_dict_name = "sevirlr_vae.pt"
pytorch_loss_state_dict_name = "sevirlr_vae_loss.pt"


class VAESEVIRPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 accumulate_grad_batches: int = 1,
                 oc_file: str = None,
                 save_dir: str = None):
        super(VAESEVIRPLModule, self).__init__()
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)

        self.torch_nn_module = AutoencoderKL(
            down_block_types=model_cfg["down_block_types"],
            in_channels=model_cfg["in_channels"],
            sample_size=model_cfg["sample_size"],  # not used
            block_out_channels=model_cfg["block_out_channels"],
            act_fn=model_cfg["act_fn"],
            latent_channels=model_cfg["latent_channels"],
            up_block_types=model_cfg["up_block_types"],
            norm_num_groups=model_cfg["norm_num_groups"],
            layers_per_block=model_cfg["layers_per_block"],
            out_channels=model_cfg["out_channels"], )
        loss_cfg = model_cfg["loss"]
        self.loss = LPIPSWithDiscriminator(
            disc_start=loss_cfg["disc_start"],
            kl_weight=loss_cfg["kl_weight"],
            disc_weight=loss_cfg["disc_weight"],
            perceptual_weight=loss_cfg["perceptual_weight"],
            disc_in_channels=loss_cfg["disc_in_channels"],)

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.layout = oc.layout.layout
        self.channel_axis = self.layout.find("C")
        self.batch_axis = self.layout.find("N")
        self.t_axis = self.layout.find("T")
        self.h_axis = self.layout.find("H")
        self.w_axis = self.layout.find("W")
        self.channels = model_cfg["data_channels"]
        # optimization
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
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

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(default_exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
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
        cfg.img_height = 128
        cfg.img_width = 128
        cfg.layout = "NHWC"
        return cfg

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        cfg.data_channels = 4
        # from stable-diffusion-v1-5
        cfg.down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D']
        cfg.in_channels = cfg.data_channels
        cfg.sample_size = 512  # not used
        cfg.block_out_channels = [128, 256, 512, 512]
        cfg.act_fn = 'silu'
        cfg.latent_channels = 4
        cfg.up_block_types = ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D']
        cfg.norm_num_groups = 32
        cfg.layers_per_block = 2
        cfg.out_channels = cfg.data_channels

        cfg.loss = OmegaConf.create()
        cfg.loss.disc_start = 50001
        cfg.loss.kl_weight = 1e-6
        cfg.loss.disc_weight = 0.5
        cfg.loss.perceptual_weight = 1.0
        cfg.loss.disc_in_channels = cfg.data_channels
        return cfg

    @classmethod
    def get_dataset_config(cls):
        cfg = OmegaConf.create()
        cfg.dataset_name = "sevirlr"
        cfg.img_height = 128
        cfg.img_width = 128
        cfg.in_len = 0
        cfg.out_len = 1
        cfg.seq_len = 1
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

        cfg.method = "adam"
        cfg.lr = 1E-3
        cfg.wd = 1E-5
        cfg.betas = (0.5, 0.9)
        cfg.gradient_clip_val = 1.0
        cfg.max_epochs = 50
        # scheduler
        cfg.warmup_percentage = 0.2
        cfg.lr_scheduler_mode = "cosine"
        cfg.min_lr_ratio = 1.0E-3
        cfg.warmup_min_lr_ratio = 0.0
        # early stopping
        cfg.monitor = "val/total_loss"
        cfg.early_stop = False
        cfg.early_stop_mode = "min"
        cfg.early_stop_patience = 5
        cfg.save_top_k = 1
        return cfg

    @staticmethod
    def get_logging_config():
        cfg = OmegaConf.create()
        cfg.logging_prefix = "SEVIRLR"
        cfg.monitor_lr = True
        cfg.monitor_device = False
        cfg.track_grad_norm = -1
        cfg.use_wandb = False
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
        cfg.num_vis = 10
        return cfg

    def configure_optimizers(self):
        optim_cfg = self.oc.optim
        lr = optim_cfg.lr
        betas = optim_cfg.betas
        opt_ae = torch.optim.Adam(list(self.torch_nn_module.encoder.parameters()) +
                                  list(self.torch_nn_module.decoder.parameters()) +
                                  list(self.torch_nn_module.quant_conv.parameters()) +
                                  list(self.torch_nn_module.post_quant_conv.parameters()),
                                  lr=lr, betas=betas)
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=betas)

        warmup_iter = int(np.round(optim_cfg.warmup_percentage * self.total_num_steps))
        if optim_cfg.lr_scheduler_mode == 'none':
            return [{"optimizer": opt_ae}, {"optimizer": opt_disc}]
        else:
            if optim_cfg.lr_scheduler_mode == 'cosine':
                # generator
                warmup_scheduler_ae = LambdaLR(
                    opt_ae,
                    lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                            min_lr_ratio=optim_cfg.warmup_min_lr_ratio))
                cosine_scheduler_ae = CosineAnnealingLR(
                    opt_ae,
                    T_max=(self.total_num_steps - warmup_iter),
                    eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr)
                lr_scheduler_ae = SequentialLR(
                    opt_ae,
                    schedulers=[warmup_scheduler_ae, cosine_scheduler_ae],
                    milestones=[warmup_iter])
                lr_scheduler_config_ae = {
                    'scheduler': lr_scheduler_ae,
                    'interval': 'step',
                    'frequency': 1, }
                # discriminator
                warmup_scheduler_disc = LambdaLR(
                    opt_disc,
                    lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                            min_lr_ratio=optim_cfg.warmup_min_lr_ratio))
                cosine_scheduler_disc = CosineAnnealingLR(
                    opt_disc,
                    T_max=(self.total_num_steps - warmup_iter),
                    eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr)
                lr_scheduler_disc = SequentialLR(
                    opt_disc,
                    schedulers=[warmup_scheduler_disc, cosine_scheduler_disc],
                    milestones=[warmup_iter])
                lr_scheduler_config_disc = {
                    'scheduler': lr_scheduler_disc,
                    'interval': 'step',
                    'frequency': 1, }
            else:
                raise NotImplementedError
            return [
                {"optimizer": opt_ae, "lr_scheduler": lr_scheduler_config_ae},
                {"optimizer": opt_disc, "lr_scheduler": lr_scheduler_config_disc},
            ]

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
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
            num_sanity_val_steps=self.oc.trainer.num_sanity_val_steps,
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            # strategy="ddp",
            strategy=DDPStrategy(find_unused_parameters=self.oc.trainer.find_unused_parameters),
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            # gradient_clip_val=self.oc.optim.gradient_clip_val,  # disabled in manual optimization
            # NVIDIA amp
            precision=self.oc.trainer.precision,
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

    def get_last_layer(self):
        return self.torch_nn_module.decoder.conv_out.weight

    def get_input(self, batch):
        target_bchw = rearrange(batch, "b 1 h w c -> b c h w").contiguous()
        mask = None
        return target_bchw, mask

    def forward(self, target_bchw, sample_posterior=True):
        pred_bchw, posterior =  self.torch_nn_module(
            sample=target_bchw,
            sample_posterior=sample_posterior,
            return_posterior=True)
        return pred_bchw, posterior

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()

        target_bchw, _ = self.get_input(batch=batch)
        pred_bchw, posterior = self(target_bchw)
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if self.current_epoch % self.oc.trainer.check_val_every_n_epoch == 0 \
                and self.local_rank == 0:
            self.save_vis_step_end(
                data_idx=data_idx,
                target=target_bchw.detach().float().cpu().numpy(),
                pred=pred_bchw.detach().float().cpu().numpy(),
                mode="train", )

        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(target_bchw, pred_bchw, posterior, optimizer_idx=0, global_step=self.global_step,
                                        mask=None, last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)
        g_opt.zero_grad()
        self.manual_backward(aeloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(g_opt, gradient_clip_val=self.oc.optim.gradient_clip_val, gradient_clip_algorithm="norm")
            g_opt.step()
            g_sch.step()

        # train the discriminator
        discloss, log_dict_disc = self.loss(target_bchw, pred_bchw, posterior, optimizer_idx=1, global_step=self.global_step,
                                            mask=None, last_layer=self.get_last_layer(), split="train")
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)

        d_opt.zero_grad()
        self.manual_backward(discloss)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(d_opt, gradient_clip_val=self.oc.optim.gradient_clip_val, gradient_clip_algorithm="norm")
            d_opt.step()
            d_sch.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.val_example_data_idx_list:
            target_bchw, _ = self.get_input(batch=batch)
            pred_bchw, posterior = self(target_bchw)
            target_bchw = target_bchw.contiguous()
            pred_bchw = pred_bchw.contiguous()
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    target=target_bchw.detach().float().cpu().numpy(),
                    pred=pred_bchw.detach().float().cpu().numpy(),
                    mode="val", )
            aeloss, log_dict_ae = self.loss(target_bchw, pred_bchw, posterior, 0, self.global_step,
                                            mask=None, last_layer=self.get_last_layer(), split="val")
            discloss, log_dict_disc = self.loss(target_bchw, pred_bchw, posterior, 1, self.global_step,
                                                mask=None, last_layer=self.get_last_layer(), split="val")
            self.log("val/rec_loss", log_dict_ae["val/rec_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.valid_mse(pred_bchw, target_bchw)
            self.valid_mae(pred_bchw, target_bchw)

    def on_validation_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()

        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        micro_batch_size = batch.shape[self.batch_axis]
        data_idx = int(batch_idx * micro_batch_size)
        if not self.eval_example_only or data_idx in self.test_example_data_idx_list:
            target_bchw, _ = self.get_input(batch=batch)
            pred_bchw, posterior = self(target_bchw)
            target_bchw = target_bchw.contiguous()
            pred_bchw = pred_bchw.contiguous()
            if self.local_rank == 0:
                self.save_vis_step_end(
                    data_idx=data_idx,
                    target=target_bchw.detach().float().cpu().numpy(),
                    pred=pred_bchw.detach().float().cpu().numpy(),
                    mode="test", )
            aeloss, log_dict_ae = self.loss(target_bchw, pred_bchw, posterior, 0, self.global_step,
                                            mask=None, last_layer=self.get_last_layer(), split="test")
            discloss, log_dict_disc = self.loss(target_bchw, pred_bchw, posterior, 1, self.global_step,
                                                mask=None, last_layer=self.get_last_layer(), split="test")
            self.log("test/rec_loss", log_dict_ae["test/rec_loss"],
                     prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.test_mse(pred_bchw, target_bchw)
            self.test_mae(pred_bchw, target_bchw)

    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()

        self.log('test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_mse.reset()
        self.test_mae.reset()

    def save_vis_step_end(
            self,
            data_idx: int,
            target: np.ndarray,
            pred: np.ndarray,
            mode: str = "train",
            prefix: str = ""):
        r"""
        Parameters
        ----------
        data_idx
        target, pred:   np.ndarray
            Shape = (N, C, H, W), actually (T, 1, H, W)
        mode:   str
        prefix: str
        """
        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if data_idx in example_data_idx_list:
                save_name = f"{prefix}{mode}_epoch_{self.current_epoch}_data_{data_idx}.png"
                num_vis = min(target.shape[0], self.oc.eval.num_vis)
                seq_list = [
                    target[:num_vis].squeeze(1),
                    pred[:num_vis].squeeze(1),
                ]
                label_list = [
                    "Target",
                    f"{self.oc.logging.logging_prefix}",
                ]
                vis_sevir_seq(
                    save_path=os.path.join(self.example_save_dir, save_name),
                    seq=seq_list,
                    label=label_list,
                    plot_stride=1, fs=20, label_rotation=90)

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
    parser.add_argument('--gpus', default=1, type=int)
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
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "vae_sevirlr_v1.yaml"))
        download_pretrained_weights(ckpt_name=pretrained_sevirlr_vae_name,
                                    save_dir=default_pretrained_vae_dir,
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
        dataset_cfg = OmegaConf.to_object(VAESEVIRPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
        float32_matmul_precision = "high"
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed_everything(seed, workers=True)
    dm = VAESEVIRPLModule.get_sevir_datamodule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=8,)
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = VAESEVIRPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = VAESEVIRPLModule(
        total_num_steps=total_num_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        save_dir=args.save,
        oc_file=args.cfg)
    trainer_kwargs = pl_module.set_trainer_kwargs(devices=args.gpus)
    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        vae_ckpt_path = os.path.join(default_pretrained_vae_dir,
                                     pretrained_sevirlr_vae_name)
        state_dict = torch.load(vae_ckpt_path,
                                map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)
        trainer.test(model=pl_module,
                     datamodule=dm)
    elif args.test:
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
        # save state_dict of VAE and discriminator
        pl_ckpt = pl_load(path_or_url=trainer.checkpoint_callback.best_model_path,
                          map_location=torch.device("cpu"))
        # state_dict = pl_ckpt["state_dict"]  # pl 1.x
        state_dict = pl_ckpt
        vae_key = "torch_nn_module."
        vae_state_dict = OrderedDict()
        loss_key = "loss."
        loss_state_dict = OrderedDict()
        unexpected_dict = OrderedDict()
        for key, val in state_dict.items():
            if key.startswith(vae_key):
                vae_state_dict[key[len(vae_key):]] = val
            elif key.startswith(loss_key):
                loss_state_dict[key[len(loss_key):]] = val
            else:
                unexpected_dict[key] = val
        torch.save(vae_state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        torch.save(loss_state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_loss_state_dict_name))
        # test
        trainer.test(ckpt_path="best",
                     datamodule=dm)


if __name__ == "__main__":
    main()
