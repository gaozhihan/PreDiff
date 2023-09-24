from typing import Dict, Any
import torch
from .alignment_pl import get_sample_align_fn
from .models import NoisyCuboidTransformerEncoder


class SEVIRAvgIntensityAlignment():

    def __init__(self,
                 alignment_type: str = "avg_x",
                 guide_scale: float = 1.0,
                 model_type: str = "cuboid",
                 model_args: Dict[str, Any] = None,
                 model_ckpt_path: str = None,
                 ):
        r"""

        Parameters
        ----------
        alignment_type: str
        guide_scale:    float
        model_type: str
        model_args: Dict[str, Any]
        model_ckpt_path:    str
            if not None, load the model from the checkpoint
        """
        super().__init__()
        assert alignment_type in ["avg_x", ], f"alignment_type {alignment_type} is not supported"
        self.alignment_type = alignment_type
        self.guide_scale = guide_scale
        if model_args is None:
            model_args = {}
        if model_type == "cuboid":
            self.model = NoisyCuboidTransformerEncoder(**model_args)
        else:
            raise NotImplementedError(f"model_type={model_type} is not implemented")
        if model_ckpt_path is not None:
            self.model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))

    @classmethod
    def model_objective(cls, x, y=None, **kwargs):
        r"""
        Parameters
        ----------
        x:  torch.Tensor
            shape = (b t h w c)
        Returns
        -------
        avg:    torch.Tensor
            shape = (b t 1)
        """
        b, t, _, _, _ = x.shape
        return torch.mean(x, dim=[2, 3, 4], keepdim=False).unsqueeze(-1)

    def alignment_fn(self, zt, t, y=None, zc=None, **kwargs):
        r"""
        transform the learned model to the final guidance \mathcal{F}.

        Parameters
        ----------
        zt: torch.Tensor
            noisy latent z
        t:  torch.Tensor
            timestamp
        y:  torch.Tensor
            context sequence in pixel space
        zc: torch.Tensor
            encoded context sequence in latente space
        kwargs: Dict[str, Any]
            auxiliary knowledge for guided generation
            `avg_x_gt`: float is required.
        Returns
        -------
        ret:    torch.Tensor
        """
        pred = self.model(zt, t, zc=zc, y=y, **kwargs)
        if self.alignment_type == "avg_x":
            target = kwargs.get("avg_x_gt")
        else:
            raise NotImplementedError
        pred = pred.mean(dim=1)  # b t 1 -> b 1
        ret = torch.linalg.vector_norm(pred - target, ord=2)
        return ret

    def get_mean_shift(self, zt, t, y=None, zc=None, **kwargs):
        r"""
        Parameters
        ----------
        zt: torch.Tensor
            noisy latent z
        t:  torch.Tensor
            timestamp
        y:  torch.Tensor
            context sequence in pixel space
        zc: torch.Tensor
            encoded context sequence in latente space
        Returns
        -------
        ret:    torch.Tensor
            \nabla_zt U
        """
        grad_fn = get_sample_align_fn(self.alignment_fn)
        grad = grad_fn(zt, t, y=y, zc=zc, **kwargs)
        return self.guide_scale * grad
