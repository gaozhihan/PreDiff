import torch
from torch import nn
from torch.utils import checkpoint

from .utils import conv_nd, apply_initialization
from .openaimodel import Upsample, Downsample


class TimeEmbedLayer(nn.Module):

    def __init__(self,
                 base_channels,
                 time_embed_channels,
                 linear_init_mode="0"):
        super(TimeEmbedLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(base_channels, time_embed_channels),
            nn.SiLU(),
            nn.Linear(time_embed_channels, time_embed_channels),
        )
        self.linear_init_mode = linear_init_mode

    def forward(self, x):
        return self.layer(x)

    def reset_parameters(self):
        apply_initialization(self.layer[0], linear_mode=self.linear_init_mode)
        apply_initialization(self.layer[2], linear_mode=self.linear_init_mode)


class TimeEmbedResBlock(nn.Module):
    r"""
    Code is adapted from https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py

    Modifications:
    1. Change GroupNorm32 to use arbitrary `num_groups`.
    2. Add method `self.reset_parameters()`.
    3. Use gradient checkpoint from PyTorch instead of the stable diffusion implementation https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/util.py#L102.
    4. If no input time embed, it degrades to res block.
    """
    def __init__(
            self,
            channels,
            dropout,
            emb_channels=None,
            out_channels=None,
            use_conv=False,
            use_embed=True,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            norm_groups=32,
    ):
        r"""
        Parameters
        ----------
        channels
        dropout
        emb_channels
        out_channels
        use_conv
        use_embed:  bool
            include `emb` as input in `self.forward()`
        use_scale_shift_norm:   bool
            take effect only when `use_embed == True`
        dims
        use_checkpoint
        up
        down
        norm_groups
        """
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.use_embed = use_embed
        if use_embed:
            assert isinstance(emb_channels, int)
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_checkpoint:
            warnings.warn("use_checkpoint is not supported yet.")
            use_checkpoint = False
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups if channels % norm_groups == 0 else channels,
                         num_channels=channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if use_embed:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    in_features=emb_channels,
                    out_features=2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups if self.out_channels % norm_groups == 0 else  self.out_channels,
                         num_channels=self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        self.reset_parameters()

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Parameters
        ----------
        x: an [N x C x ...] Tensor of features.
        emb: an [N x emb_channels] Tensor of timestep embeddings.

        Returns
        -------
        out: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if self.use_embed:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
        else:
            h = self.out_layers(h)
        return self.skip_connection(x) + h

    def reset_parameters(self):
        for m in self.modules():
            apply_initialization(m)
        for p in self.out_layers[-1].parameters():
            nn.init.zeros_(p)
