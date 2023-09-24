from typing import Callable
from torch import nn
from torch.nn import functional as F


def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda


def get_loss_fn(loss: str = "l2") -> Callable:
    if loss in ("l2", "mse"):
        return F.mse_loss
    elif loss in ("l1", "mae"):
        return F.l1_loss


def disabled_train(self):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def disable_train(model: nn.Module):
    r"""
    Disable training to avoid error when used in pl.LightningModule
    """
    model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False
    return model
