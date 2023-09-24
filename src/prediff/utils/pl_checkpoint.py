from typing import IO, Union, Callable
from collections import OrderedDict
import torch
from lightning_fabric.utilities.cloud_io import _load
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.utilities.migration import pl_legacy_patch
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint


def pl_load(
    path_or_url: Union[IO, _PATH],
    map_location: _MAP_LOCATION_TYPE = None,
) -> OrderedDict[str, torch.Tensor]:
    r"""
    Load the `state_dict` only from a PyTorch-Lightning checkpoint.
    Code is adopted from https://github.com/Lightning-AI/lightning/blob/255b18823e7da265e0e2e3996f55dcd0f78e9f3e/src/lightning/pytorch/core/saving.py
    """
    with pl_legacy_patch():
        checkpoint = _load(path_or_url, map_location=map_location)
    # convert legacy checkpoints to the new format
    checkpoint = _pl_migrate_checkpoint(
        checkpoint, checkpoint_path=(path_or_url if isinstance(path_or_url, _PATH) else None)
    )
    return checkpoint["state_dict"]


def pl_ckpt_to_state_dict(
    checkpoint_path: str,
    map_location: _MAP_LOCATION_TYPE = None,
    key_fn: Callable = lambda x: x,
):
    r"""
    Parameters
    ----------
    checkpoint_path:    str
    map_location:   _MAP_LOCATION_TYPE
        A function, torch.device, string or a dict specifying how to remap storage locations.
        The same as the arg `map_location` in `torch.load()`.
    key_fn: Callable
        A function to map the keys in the loaded checkpoint to the desired keys in the returned state_dict.

    Returns
    -------
    state_dict: OrderedDict
    """
    if map_location is None:
        map_location = lambda storage, loc: storage
    pl_ckpt_state_dict = pl_load(checkpoint_path, map_location=map_location)
    state_dict = {key_fn(key): val for key, val in pl_ckpt_state_dict.items()}
    return state_dict
