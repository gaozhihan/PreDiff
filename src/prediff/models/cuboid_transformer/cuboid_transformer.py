"""Code is adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/a01ea56de4baddc5b381757e5d789f7f1efdcffe/src/earthformer/cuboid_transformer/cuboid_transformer.py"""
from typing import Sequence, Union
import warnings
from functools import lru_cache
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from .cuboid_transformer_patterns import CuboidSelfAttentionPatterns, CuboidCrossAttentionPatterns
from ..utils import (
    get_activation, get_norm_layer,
    _generalize_padding, _generalize_unpadding,
    apply_initialization, round_to, WrapIdentity)


class PosEmbed(nn.Module):

    def __init__(self, embed_dim, maxT, maxH, maxW, typ='t+h+w'):
        r"""
        Parameters
        ----------
        embed_dim
        maxT
        maxH
        maxW
        typ
            The type of the positional embedding.
            - t+h+w:
                Embed the spatial position to embeddings
            - t+hw:
                Embed the spatial position to embeddings
        """
        super(PosEmbed, self).__init__()
        self.typ = typ

        assert self.typ in ['t+h+w', 't+hw']
        self.maxT = maxT
        self.maxH = maxH
        self.maxW = maxW
        self.embed_dim = embed_dim
        # spatiotemporal learned positional embedding
        if self.typ == 't+h+w':
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.H_embed = nn.Embedding(num_embeddings=maxH, embedding_dim=embed_dim)
            self.W_embed = nn.Embedding(num_embeddings=maxW, embedding_dim=embed_dim)

            # nn.init.trunc_normal_(self.T_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.H_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.W_embed.weight, std=0.02)
        elif self.typ == 't+hw':
            self.T_embed = nn.Embedding(num_embeddings=maxT, embedding_dim=embed_dim)
            self.HW_embed = nn.Embedding(num_embeddings=maxH * maxW, embedding_dim=embed_dim)
            # nn.init.trunc_normal_(self.T_embed.weight, std=0.02)
            # nn.init.trunc_normal_(self.HW_embed.weight, std=0.02)
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m, embed_mode="0")

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C)

        Returns
        -------
        out
            Return the x + positional embeddings
        """
        _, T, H, W, _ = x.shape
        t_idx = torch.arange(T, device=x.device)  # (T, C)
        h_idx = torch.arange(H, device=x.device)  # (H, C)
        w_idx = torch.arange(W, device=x.device)  # (W, C)
        if self.typ == 't+h+w':
            return x + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim)\
                     + self.H_embed(h_idx).reshape(1, H, 1, self.embed_dim)\
                     + self.W_embed(w_idx).reshape(1, 1, W, self.embed_dim)
        elif self.typ == 't+hw':
            spatial_idx = h_idx.unsqueeze(-1) * self.maxW + w_idx
            return x + self.T_embed(t_idx).reshape(T, 1, 1, self.embed_dim) + self.HW_embed(spatial_idx)
        else:
            raise NotImplementedError


class PositionwiseFFN(nn.Module):
    """The Position-wise FFN layer used in Transformer-like architectures

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    Also, if we use gated projection. We will use
        fc1_1 * act(fc1_2(data)) to map the data
    """
    def __init__(self,
                 units: int = 512,
                 hidden_size: int = 2048,
                 activation_dropout: float = 0.0,
                 dropout: float = 0.1,
                 gated_proj: bool = False,
                 activation='relu',
                 normalization: str = 'layer_norm',
                 layer_norm_eps: float = 1E-5,
                 pre_norm: bool = False,
                 linear_init_mode="0",
                 ffn2_linear_init_mode="2",
                 norm_init_mode="0",
                 ):
        """
        Parameters
        ----------
        units
        hidden_size
        activation_dropout
        dropout
        activation
        normalization
            layer_norm or no_norm
        layer_norm_eps
        pre_norm
            Pre-layer normalization as proposed in the paper:
            "[ACL2018] The Best of Both Worlds: Combining Recent Advances in
             Neural Machine Translation"
            This will stabilize the training of Transformers.
            You may also refer to
            "[Arxiv2020] Understanding the Difficulty of Training Transformers"
        """
        super().__init__()
        # initialization
        self.linear_init_mode = linear_init_mode
        self.ffn2_linear_init_mode = ffn2_linear_init_mode
        self.norm_init_mode = norm_init_mode

        self._pre_norm = pre_norm
        self._gated_proj = gated_proj
        self._kwargs = OrderedDict([
            ('units', units),
            ('hidden_size', hidden_size),
            ('activation_dropout', activation_dropout),
            ('activation', activation),
            ('dropout', dropout),
            ('normalization', normalization),
            ('layer_norm_eps', layer_norm_eps),
            ('gated_proj', gated_proj),
            ('pre_norm', pre_norm)
        ])
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_dropout_layer = nn.Dropout(activation_dropout)
        self.ffn_1 = nn.Linear(in_features=units, out_features=hidden_size,
                               bias=True)
        if self._gated_proj:
            self.ffn_1_gate = nn.Linear(in_features=units,
                                        out_features=hidden_size,
                                        bias=True)
        self.activation = get_activation(activation)
        self.ffn_2 = nn.Linear(in_features=hidden_size, out_features=units,
                               bias=True)
        self.layer_norm = get_norm_layer(norm_type=normalization,
                                         in_channels=units,
                                         epsilon=layer_norm_eps)
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.ffn_1,
                             linear_mode=self.linear_init_mode)
        if self._gated_proj:
            apply_initialization(self.ffn_1_gate,
                                 linear_mode=self.linear_init_mode)
        apply_initialization(self.ffn_2,
                             linear_mode=self.ffn2_linear_init_mode)
        apply_initialization(self.layer_norm,
                             norm_mode=self.norm_init_mode)

    def forward(self, data):
        """

        Parameters
        ----------
        data :
            Shape (B, seq_length, C_in)

        Returns
        -------
        out :
            Shape (B, seq_length, C_out)
        """
        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        if self._gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out


class PatchMerging3D(nn.Module):
    """ Patch Merging Layer"""
    def __init__(self,
                 dim,
                 out_dim=None,
                 downsample=(1, 2, 2),
                 norm_layer='layer_norm',
                 padding_type='nearest',
                 linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
            Number of input channels.
        downsample
            downsample factor
        norm_layer
            The normalization layer
        """
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.dim = dim
        if out_dim is None:
            out_dim = max(downsample) * dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.padding_type = padding_type
        self.reduction = nn.Linear(downsample[0] * downsample[1] * downsample[2] * dim,
                                   out_dim, bias=False)
        self.norm = get_norm_layer(norm_layer, in_channels=downsample[0] * downsample[1] * downsample[2] * dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 linear_mode=self.linear_init_mode,
                                 norm_mode=self.norm_init_mode)

    def get_out_shape(self, data_shape):
        T, H, W, C_in = data_shape
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        return (T + pad_t) // self.downsample[0], (H + pad_h) // self.downsample[1], (W + pad_w) // self.downsample[2],\
               self.out_dim

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Input feature, tensor size (B, T, H, W, C).

        Returns
        -------
        out
            Shape (B, T // downsample[0], H // downsample[1], W // downsample[2], out_dim)
        """
        B, T, H, W, C = x.shape

        # padding
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        if pad_h or pad_h or pad_w:
            T += pad_t
            H += pad_h
            W += pad_w
            x = _generalize_padding(x, pad_t, pad_h, pad_w, padding_type=self.padding_type)

        x = x.reshape((B,
                       T // self.downsample[0], self.downsample[0],
                       H // self.downsample[1], self.downsample[1],
                       W // self.downsample[2], self.downsample[2], C)) \
             .permute(0, 1, 3, 5, 2, 4, 6, 7) \
             .reshape(B, T // self.downsample[0], H // self.downsample[1], W // self.downsample[2],
                      self.downsample[0] * self.downsample[1] * self.downsample[2] * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Upsample3DLayer(nn.Module):
    """Upsampling based on nn.UpSampling and Conv3x3.

    If the temporal dimension remains the same:
        x --> interpolation-2d (nearest) --> conv3x3(dim, out_dim)
    Else:
        x --> interpolation-3d (nearest) --> conv3x3x3(dim, out_dim)

    """
    def __init__(self,
                 dim,
                 out_dim,
                 target_size,
                 temporal_upsample=False,
                 kernel_size=3,
                 layout='THWC',
                 conv_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
        out_dim
        target_size
            Size of the output tensor. Will be a tuple/list that contains T_new, H_new, W_new
        temporal_upsample
            Whether the temporal axis will go through upsampling.
        kernel_size
            The kernel size of the Conv2D layer
        layout
            The layout of the inputs
        """
        super(Upsample3DLayer, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.target_size = target_size
        self.out_dim = out_dim
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.up = nn.Upsample(size=target_size, mode='nearest')  # 3D upsampling
        else:
            self.up = nn.Upsample(size=(target_size[1], target_size[2]), mode='nearest')  # 2D upsampling
        self.conv = nn.Conv2d(in_channels=dim, out_channels=out_dim, kernel_size=(kernel_size, kernel_size),
                              padding=(kernel_size // 2, kernel_size // 2))
        assert layout in ['THWC', 'CTHW']
        self.layout = layout

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            apply_initialization(m,
                                 conv_mode=self.conv_init_mode)

    def forward(self, x):
        """

        Parameters
        ----------
        x
            Shape (B, T, H, W, C) or (B, C, T, H, W)

        Returns
        -------
        out
            Shape (B, T, H_new, W_out, C_out) or (B, C, T, H_out, W_out)
        """
        if self.layout == 'THWC':
            B, T, H, W, C = x.shape
            if self.temporal_upsample:
                x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
                return self.conv(self.up(x)).permute(0, 2, 3, 4, 1)
            else:
                assert self.target_size[0] == T
                x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B * T, C, H, W)
                x = self.up(x)
                return self.conv(x).permute(0, 2, 3, 1).reshape((B,) + self.target_size + (self.out_dim,))
        elif self.layout == 'CTHW':
            B, C, T, H, W = x.shape
            if self.temporal_upsample:
                return self.conv(self.up(x))
            else:
                assert self.output_size[0] == T
                x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
                x = x.reshape(B * T, C, H, W)
                return self.conv(self.up(x)).reshape(B, self.target_size[0], self.out_dim, self.target_size[1],
                                                     self.target_size[2]).permute(0, 2, 1, 3, 4)


def cuboid_reorder(data, cuboid_size, strategy):
    """Reorder the tensor into (B, num_cuboids, bT * bH * bW, C)

    We assume that the tensor shapes are divisible to the cuboid sizes.

    Parameters
    ----------
    data
        The input data
    cuboid_size
        The size of the cuboid
    strategy
        The cuboid strategy

    Returns
    -------
    reordered_data
        Shape will be (B, num_cuboids, bT * bH * bW, C)
        num_cuboids = T / bT * H / bH * W / bW
    """
    B, T, H, W, C = data.shape
    num_cuboids = T // cuboid_size[0] * H // cuboid_size[1] * W // cuboid_size[2]
    cuboid_volume = cuboid_size[0] * cuboid_size[1] * cuboid_size[2]
    intermediate_shape = []

    nblock_axis = []
    block_axis = []
    for i, (block_size, total_size, ele_strategy) in enumerate(zip(cuboid_size, (T, H, W), strategy)):
        if ele_strategy == 'l':
            intermediate_shape.extend([total_size // block_size, block_size])
            nblock_axis.append(2 * i + 1)
            block_axis.append(2 * i + 2)
        elif ele_strategy == 'd':
            intermediate_shape.extend([block_size, total_size // block_size])
            nblock_axis.append(2 * i + 2)
            block_axis.append(2 * i + 1)
        else:
            raise NotImplementedError
    data = data.reshape((B,) + tuple(intermediate_shape) + (C, ))
    reordered_data = data.permute((0,) + tuple(nblock_axis) + tuple(block_axis) + (7,))
    reordered_data = reordered_data.reshape((B, num_cuboids, cuboid_volume, C))
    return reordered_data


def cuboid_reorder_reverse(data, cuboid_size, strategy, orig_data_shape):
    """Reverse the reordered cuboid back to the original space

    Parameters
    ----------
    data
    cuboid_size
    strategy
    orig_data_shape

    Returns
    -------
    data
        The recovered data
    """
    B, num_cuboids, cuboid_volume, C = data.shape
    T, H, W = orig_data_shape

    permutation_axis = [0]
    for i, (block_size, total_size, ele_strategy) in enumerate(zip(cuboid_size, (T, H, W), strategy)):
        if ele_strategy == 'l':
            # intermediate_shape.extend([total_size // block_size, block_size])
            permutation_axis.append(i + 1)
            permutation_axis.append(i + 4)
        elif ele_strategy == 'd':
            # intermediate_shape.extend([block_size, total_size // block_size])
            permutation_axis.append(i + 4)
            permutation_axis.append(i + 1)
        else:
            raise NotImplementedError
    permutation_axis.append(7)
    data = data.reshape(B, T // cuboid_size[0], H // cuboid_size[1], W // cuboid_size[2],
                        cuboid_size[0], cuboid_size[1], cuboid_size[2], C)
    data = data.permute(permutation_axis)
    data = data.reshape((B, T, H, W, C))
    return data


@lru_cache()
def compute_cuboid_self_attention_mask(data_shape, cuboid_size, shift_size, strategy, padding_type, device):
    """Compute the shift window attention mask

    Parameters
    ----------
    data_shape
        Should be T, H, W
    cuboid_size
        Size of the cuboid
    shift_size
        The shift size
    strategy
        The decomposition strategy
    padding_type
        Type of the padding
    device
        The device

    Returns
    -------
    attn_mask
        Mask with shape (num_cuboid, cuboid_vol, cuboid_vol)
        The padded values will always be masked. The other masks will ensure that the shifted windows
        will only attend to those in the shifted windows.
    """
    T, H, W = data_shape
    pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
    pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
    pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]
    data_mask = None
    # Prepare data mask
    if pad_t > 0  or pad_h > 0 or pad_w > 0:
        if padding_type == 'ignore':
            data_mask = torch.ones((1, T, H, W, 1), dtype=torch.bool, device=device)
            data_mask = F.pad(data_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    else:
        data_mask = torch.ones((1, T + pad_t, H + pad_h, W + pad_w, 1), dtype=torch.bool, device=device)
    if any(i > 0 for i in shift_size):
        if padding_type == 'ignore':
            data_mask = torch.roll(data_mask, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    if padding_type == 'ignore':
        # (1, num_cuboids, cuboid_volume, 1)
        data_mask = cuboid_reorder(data_mask, cuboid_size, strategy=strategy)
        data_mask = data_mask.squeeze(-1).squeeze(0)  # (num_cuboid, cuboid_volume)
    # Prepare mask based on index
    shift_mask = torch.zeros((1, T + pad_t, H + pad_h, W + pad_w, 1), device=device)  # 1 T H W 1
    cnt = 0
    for t in slice(-cuboid_size[0]), slice(-cuboid_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-cuboid_size[1]), slice(-cuboid_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-cuboid_size[2]), slice(-cuboid_size[2], -shift_size[2]), slice(-shift_size[2], None):
                shift_mask[:, t, h, w, :] = cnt
                cnt += 1
    shift_mask = cuboid_reorder(shift_mask, cuboid_size, strategy=strategy)
    shift_mask = shift_mask.squeeze(-1).squeeze(0)  # num_cuboids, cuboid_volume
    attn_mask = (shift_mask.unsqueeze(1) - shift_mask.unsqueeze(2)) == 0  # num_cuboids, cuboid_volume, cuboid_volume
    if padding_type == 'ignore':
        attn_mask = data_mask.unsqueeze(1) * data_mask.unsqueeze(2) * attn_mask
    return attn_mask


def masked_softmax(att_score, mask, axis: int = -1):
    """Ignore the masked elements when calculating the softmax.
     The mask can be broadcastable.

    Parameters
    ----------
    att_score
        Shape (..., length, ...)
    mask
        Shape (..., length, ...)
        1 --> The element is not masked
        0 --> The element is masked
    axis
        The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]

    Returns
    -------
    att_weights
        Shape (..., length, ...)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        if att_score.dtype == torch.float16:
            att_score = att_score.masked_fill(torch.logical_not(mask), -1E4)
        else:
            att_score = att_score.masked_fill(torch.logical_not(mask), -1E18)
        att_weights = torch.softmax(att_score, dim=axis) * mask
    else:
        att_weights = torch.softmax(att_score, dim=axis)
    return att_weights


def update_cuboid_size_shift_size(data_shape, cuboid_size, shift_size, strategy):
    """Update the

    Parameters
    ----------
    data_shape
        The shape of the data
    cuboid_size
        Size of the cuboid
    shift_size
        Size of the shift
    strategy
        The strategy of attention

    Returns
    -------
    new_cuboid_size
        Size of the cuboid
    new_shift_size
        Size of the shift
    """
    new_cuboid_size = list(cuboid_size)
    new_shift_size = list(shift_size)
    for i in range(len(data_shape)):
        if strategy[i] == 'd':
            new_shift_size[i] = 0
        if data_shape[i] <= cuboid_size[i]:
            new_cuboid_size[i] = data_shape[i]
            new_shift_size[i] = 0
    return tuple(new_cuboid_size), tuple(new_shift_size)


class CuboidSelfAttentionLayer(nn.Module):
    """Implements the cuboid self attention.

    The idea of Cuboid Self Attention is to divide the input tensor (T, H, W) into several non-overlapping cuboids.
    We apply self-attention inside each cuboid and all cuboid-level self attentions are executed in parallel.

    We adopt two mechanisms for decomposing the input tensor into cuboids:

    1) local:
        We group the tensors within a local window, e.g., X[t:(t+b_t), h:(h+b_h), w:(w+b_w)]. We can also apply the
        shifted window strategy proposed in "[ICCV2021] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
    2) dilated:
        Inspired by the success of dilated convolution "[ICLR2016] Multi-Scale Context Aggregation by Dilated Convolutions",
         we split the tensor with dilation factors that are tied to the size of the cuboid. For example, for a cuboid that has width `b_w`,
         we sample the elements starting from 0 as 0, w / b_w, 2 * w / b_w, ..., (b_w - 1) * w / b_w.

    The cuboid attention can be viewed as a generalization of the attention mechanism proposed in Video Swin Transformer, https://arxiv.org/abs/2106.13230.
    The computational complexity of CuboidAttention can be simply calculated as O(T H W * b_t b_h b_w). To cover multiple correlation patterns,
    we are able to combine multiple CuboidAttention layers with different configurations such as cuboid size, shift size, and local / global decomposing strategy.

    In addition, it is straight-forward to extend the cuboid attention to other types of spatiotemporal data that are not described
    as regular tensors. We need to define alternative approaches to partition the data into "cuboids".

    In addition, inspired by "[NeurIPS2021] Do Transformers Really Perform Badly for Graph Representation?",
     "[NeurIPS2020] Big Bird: Transformers for Longer Sequences", "[EMNLP2021] Longformer: The Long-Document Transformer", we keep
     $K$ global vectors to record the global status of the spatiotemporal system. These global vectors will attend to the whole tensor and
     the vectors inside each individual cuboids will also attend to the global vectors so that they can peep into the global status of the system.

    """
    def __init__(self,
                 dim,
                 num_heads,
                 cuboid_size=(2, 7, 7),
                 shift_size=(0, 0, 0),
                 strategy=('l', 'l', 'l'),
                 padding_type='ignore',
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 use_final_proj=True,
                 norm_layer='layer_norm',
                 use_global_vector=False,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 checkpoint_level=True,
                 use_relative_pos=True,
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="2",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
            The dimension of the input tensor
        num_heads
            The number of heads
        cuboid_size
            The size of each cuboid
        shift_size
            The size for shifting the windows.
        strategy
            The decomposition strategy of the tensor. 'l' stands for local and 'd' stands for dilated.
        padding_type
            The type of padding.
        qkv_bias
            Whether to enable bias in calculating qkv attention
        qk_scale
            Whether to enable scale factor when calculating the attention.
        attn_drop
            The attention dropout
        proj_drop
            The projection dropout
        use_final_proj
            Whether to use the final projection or not
        norm_layer
            The normalization layer
        use_global_vector
            Whether to use the global vector or not.
        use_global_self_attn
            Whether to do self attention among global vectors
        separate_global_qkv
            Whether to different network to calc q_global, k_global, v_global
        global_dim_ratio
            The dim (channels) of global vectors is `global_dim_ratio*dim`.
        checkpoint_level
            Whether to enable gradient checkpointing.
        """
        super(CuboidSelfAttentionLayer, self).__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.cuboid_size = cuboid_size
        self.shift_size = shift_size
        self.strategy = strategy
        self.padding_type = padding_type
        self.use_final_proj = use_final_proj
        self.use_relative_pos = use_relative_pos
        # global vectors
        self.use_global_vector = use_global_vector
        self.use_global_self_attn = use_global_self_attn
        self.separate_global_qkv = separate_global_qkv
        if global_dim_ratio != 1:
            assert separate_global_qkv == True, \
                f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio

        assert self.padding_type in ['ignore', 'zeros', 'nearest']
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if use_relative_pos:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * cuboid_size[0] - 1) * (2 * cuboid_size[1] - 1) * (2 * cuboid_size[2] - 1), num_heads))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

            coords_t = torch.arange(self.cuboid_size[0])
            coords_h = torch.arange(self.cuboid_size[1])
            coords_w = torch.arange(self.cuboid_size[2])
            coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w))  # 3, Bt, Bh, Bw

            coords_flatten = torch.flatten(coords, 1)  # 3, Bt*Bh*Bw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Bt*Bh*Bw, Bt*Bh*Bw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Bt*Bh*Bw, Bt*Bh*Bw, 3
            relative_coords[:, :, 0] += self.cuboid_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.cuboid_size[1] - 1
            relative_coords[:, :, 2] += self.cuboid_size[2] - 1

            relative_coords[:, :, 0] *= (2 * self.cuboid_size[1] - 1) * (2 * self.cuboid_size[2] - 1)
            relative_coords[:, :, 1] *= (2 * self.cuboid_size[2] - 1)
            relative_position_index = relative_coords.sum(-1)  # shape is (cuboid_volume, cuboid_volume)
            self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_global_vector:
            if self.separate_global_qkv:
                self.l2g_q_net = nn.Linear(dim, dim, bias=qkv_bias)
                self.l2g_global_kv_net = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim * 2,
                    bias=qkv_bias)
                self.g2l_global_q_net = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim,
                    bias=qkv_bias)
                self.g2l_k_net = nn.Linear(
                    in_features=dim,
                    out_features=dim,
                    bias=qkv_bias)
                self.g2l_v_net = nn.Linear(
                    in_features=dim,
                    out_features=global_dim_ratio * dim,
                    bias=qkv_bias)
                if self.use_global_self_attn:
                    self.g2g_global_qkv_net = nn.Linear(
                        in_features=global_dim_ratio * dim,
                        out_features=global_dim_ratio * dim * 3,
                        bias=qkv_bias)
            else:
                self.global_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.global_attn_drop = nn.Dropout(attn_drop)

        if use_final_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

            if self.use_global_vector:
                self.global_proj = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=global_dim_ratio * dim)

        self.norm = get_norm_layer(norm_layer, in_channels=dim)
        if self.use_global_vector:
            self.global_vec_norm = get_norm_layer(norm_layer,
                                                  in_channels=global_dim_ratio*dim)

        self.checkpoint_level = checkpoint_level
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.qkv,
                             linear_mode=self.attn_linear_init_mode)
        if self.use_final_proj:
            apply_initialization(self.proj,
                                 linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.norm,
                             norm_mode=self.norm_init_mode)
        if self.use_global_vector:
            if self.separate_global_qkv:
                apply_initialization(self.l2g_q_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.l2g_global_kv_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_global_q_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_k_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_v_net,
                                     linear_mode=self.attn_linear_init_mode)
                if self.use_global_self_attn:
                    apply_initialization(self.g2g_global_qkv_net,
                                         linear_mode=self.attn_linear_init_mode)
            else:
                apply_initialization(self.global_qkv,
                                     linear_mode=self.attn_linear_init_mode)
            apply_initialization(self.global_vec_norm,
                                 norm_mode=self.norm_init_mode)

    def forward(self, x, global_vectors=None):
        x = self.norm(x)

        B, T, H, W, C_in = x.shape
        assert C_in == self.dim
        if self.use_global_vector:
            _, num_global, _ = global_vectors.shape
            global_vectors = self.global_vec_norm(global_vectors)

        cuboid_size, shift_size = update_cuboid_size_shift_size((T, H, W), self.cuboid_size,
                                                                self.shift_size, self.strategy)
        # Step-1: Pad the input
        pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
        pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
        pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]

        # We use generalized padding
        x = _generalize_padding(x, pad_t, pad_h, pad_w, self.padding_type)

        # Step-2: Shift the tensor based on shift window attention.

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        # Step-3: Reorder the tensor
        # (B, num_cuboids, cuboid_volume, C)
        reordered_x = cuboid_reorder(shifted_x, cuboid_size=cuboid_size, strategy=self.strategy)
        _, num_cuboids, cuboid_volume, _ = reordered_x.shape
        # Step-4: Perform self-attention
        # (num_cuboids, cuboid_volume, cuboid_volume)
        attn_mask = compute_cuboid_self_attention_mask((T, H, W), cuboid_size,
                                                       shift_size=shift_size,
                                                       strategy=self.strategy,
                                                       padding_type=self.padding_type,
                                                       device=x.device)
        head_C = C_in // self.num_heads
        qkv = self.qkv(reordered_x).reshape(B, num_cuboids, cuboid_volume, 3, self.num_heads, head_C)\
            .permute(3, 0, 4, 1, 2, 5)  # (3, B, num_heads, num_cuboids, cuboid_volume, head_C)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, num_cuboids, cuboid_volume, head_C)
        q = q * self.scale
        attn_score = q @ k.transpose(-2, -1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume)

        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:cuboid_volume, :cuboid_volume].reshape(-1)]\
                .reshape(cuboid_volume, cuboid_volume, -1)  # (cuboid_volume, cuboid_volume, num_head)
            relative_position_bias = relative_position_bias.permute(2, 0, 1)\
                .contiguous().unsqueeze(1)  # num_heads, 1, cuboid_volume, cuboid_volume
            attn_score = attn_score + relative_position_bias  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume)

        # Calculate the local to global attention
        if self.use_global_vector:
            global_head_C = self.global_dim_ratio * head_C # take effect only separate_global_qkv = True
            if self.separate_global_qkv:
                l2g_q = self.l2g_q_net(reordered_x)\
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C)\
                    .permute(0, 3, 1, 2, 4)  # (B, num_heads, num_cuboids, cuboid_volume, head_C)
                l2g_q = l2g_q * self.scale
                l2g_global_kv = self.l2g_global_kv_net(global_vectors)\
                    .reshape(B, 1, num_global, 2, self.num_heads, head_C)\
                    .permute(3, 0, 4, 1, 2, 5)  # Shape (2, B, num_heads, 1, N, head_C)
                l2g_global_k, l2g_global_v = l2g_global_kv[0], l2g_global_kv[1]
                g2l_global_q = self.g2l_global_q_net(global_vectors)\
                    .reshape(B, num_global, self.num_heads, head_C)\
                    .permute(0, 2, 1, 3)  # Shape (B, num_heads, N, head_C)
                g2l_global_q = g2l_global_q * self.scale
                # g2l_kv = self.g2l_kv_net(reordered_x)\
                #     .reshape(B, num_cuboids, cuboid_volume, 2, self.num_heads, global_head_C)\
                #     .permute(3, 0, 4, 1, 2, 5)  # (2, B, num_heads, num_cuboids, cuboid_volume, head_C)
                # g2l_k, g2l_v = g2l_kv[0], g2l_kv[1]
                g2l_k = self.g2l_k_net(reordered_x)\
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C)\
                    .permute(0, 3, 1, 2, 4)  # (B, num_heads, num_cuboids, cuboid_volume, head_C)
                g2l_v = self.g2l_v_net(reordered_x) \
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, global_head_C) \
                    .permute(0, 3, 1, 2, 4)  # (B, num_heads, num_cuboids, cuboid_volume, global_head_C)
                if self.use_global_self_attn:
                    g2g_global_qkv = self.g2g_global_qkv_net(global_vectors)\
                    .reshape(B, 1, num_global, 3, self.num_heads, global_head_C)\
                    .permute(3, 0, 4, 1, 2, 5)  # Shape (2, B, num_heads, 1, N, head_C)
                    g2g_global_q, g2g_global_k, g2g_global_v = g2g_global_qkv[0], g2g_global_qkv[1], g2g_global_qkv[2]
                    g2g_global_q = g2g_global_q.squeeze(2) * self.scale
            else:
                q_global, k_global, v_global = self.global_qkv(global_vectors)\
                    .reshape(B, 1, num_global, 3, self.num_heads, head_C)\
                    .permute(3, 0, 4, 1, 2, 5)  # Shape (3, B, num_heads, 1, N, head_C)
                q_global = q_global.squeeze(2) * self.scale
                l2g_q, g2l_k, g2l_v = q, k, v
                g2l_global_q, l2g_global_k, l2g_global_v = q_global, k_global, v_global
                if self.use_global_self_attn:
                    g2g_global_q, g2g_global_k, g2g_global_v = q_global, k_global, v_global
            l2g_attn_score = l2g_q @ l2g_global_k.transpose(-2, -1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, N)
            attn_score_l2l_l2g = torch.cat((attn_score, l2g_attn_score),
                                           dim=-1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume + N)
            attn_mask_l2l_l2g = F.pad(attn_mask, (0, num_global), "constant", 1)
            v_l_g = torch.cat((v, l2g_global_v.expand(B, self.num_heads, num_cuboids, num_global, head_C)),
                            dim=3)
            # local to local and global attention
            attn_score_l2l_l2g = masked_softmax(attn_score_l2l_l2g, mask=attn_mask_l2l_l2g)
            attn_score_l2l_l2g = self.attn_drop(attn_score_l2l_l2g)  # Shape (B, num_heads, num_cuboids, x_cuboid_volume, mem_cuboid_volume + K))
            reordered_x = (attn_score_l2l_l2g @ v_l_g).permute(0, 2, 3, 1, 4) \
                .reshape(B, num_cuboids, cuboid_volume, self.dim)
            # update global vectors
            if self.padding_type == 'ignore':
                g2l_attn_mask = torch.ones((1, T, H, W, 1), device=x.device)
                if pad_t > 0 or pad_h > 0 or pad_w > 0:
                    g2l_attn_mask = F.pad(g2l_attn_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
                if any(i > 0 for i in shift_size):
                    g2l_attn_mask = torch.roll(g2l_attn_mask, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                                               dims=(1, 2, 3))
                g2l_attn_mask = g2l_attn_mask.reshape((-1,))
            else:
                g2l_attn_mask = None
            g2l_attn_score = g2l_global_q @ g2l_k.reshape(B, self.num_heads, num_cuboids * cuboid_volume, head_C).transpose(-2, -1)  # Shape (B, num_heads, N, num_cuboids * cuboid_volume)
            if self.use_global_self_attn:
                g2g_attn_score = g2g_global_q @ g2g_global_k.squeeze(2).transpose(-2, -1)
                g2all_attn_score = torch.cat((g2l_attn_score, g2g_attn_score),
                                             dim=-1)  # Shape (B, num_heads, N, num_cuboids * cuboid_volume + N)
                if g2l_attn_mask is not None:
                    g2all_attn_mask = F.pad(g2l_attn_mask, (0, num_global), "constant", 1)
                else:
                    g2all_attn_mask = None
                new_v = torch.cat((g2l_v.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C),
                                   g2g_global_v.reshape(B, self.num_heads, num_global, global_head_C)),
                                  dim=2)
            else:
                g2all_attn_score = g2l_attn_score
                g2all_attn_mask = g2l_attn_mask
                new_v = g2l_v.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C)
            g2all_attn_score = masked_softmax(g2all_attn_score, mask=g2all_attn_mask)
            g2all_attn_score = self.global_attn_drop(g2all_attn_score)
            new_global_vector = (g2all_attn_score @ new_v).permute(0, 2, 1, 3).\
                reshape(B, num_global, self.global_dim_ratio*self.dim)
        else:
            attn_score = masked_softmax(attn_score, mask=attn_mask)
            attn_score = self.attn_drop(attn_score)  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume (+ K))
            reordered_x = (attn_score @ v).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, self.dim)

        if self.use_final_proj:
            reordered_x = self.proj_drop(self.proj(reordered_x))
            if self.use_global_vector:
                new_global_vector = self.proj_drop(self.global_proj(new_global_vector))
        # Step-5: Shift back and slice
        shifted_x = cuboid_reorder_reverse(reordered_x, cuboid_size=cuboid_size, strategy=self.strategy,
                                           orig_data_shape=(T + pad_t, H + pad_h, W + pad_w))
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = _generalize_unpadding(x, pad_t=pad_t, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type)
        if self.use_global_vector:
            return x, new_global_vector
        else:
            return x


class StackCuboidSelfAttentionBlock(nn.Module):
    """

    - "use_inter_ffn" is True
        x --> attn1 -----+-------> ffn1 ---+---> attn2 --> ... --> ffn_k --> out
           |             ^   |             ^
           |             |   |             |
           |-------------|   |-------------|
    - "use_inter_ffn" is False
        x --> attn1 -----+------> attn2 --> ... attnk --+----> ffnk ---+---> out
           |             ^   |            ^             ^  |           ^
           |             |   |            |             |  |           |
           |-------------|   |------------|   ----------|  |-----------|
    If we have enabled global memory vectors, each attention will be a

    """
    def __init__(self,
                 dim,
                 num_heads,
                 block_cuboid_size=[(4, 4, 4), (4, 4, 4)],
                 block_shift_size=[(0, 0, 0), (2, 2, 2)],
                 block_strategy=[('d', 'd', 'd'),
                                 ('l', 'l', 'l')],
                 padding_type='ignore',
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=False,
                 use_global_vector=False,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 checkpoint_level=True,
                 use_relative_pos=True,
                 use_final_proj=True,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 ffn2_linear_init_mode="2",
                 attn_proj_linear_init_mode="2",
                 norm_init_mode="0",
                 ):
        super(StackCuboidSelfAttentionBlock, self).__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.attn_proj_linear_init_mode = attn_proj_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert len(block_cuboid_size[0]) > 0 and len(block_shift_size) > 0 and len(block_strategy) > 0,\
            f'Format of the block cuboid size is not correct.' \
            f' block_cuboid_size={block_cuboid_size}'
        assert len(block_cuboid_size) == len(block_shift_size) == len(block_strategy)
        self.num_attn = len(block_cuboid_size)
        self.checkpoint_level = checkpoint_level
        self.use_inter_ffn = use_inter_ffn
        # global vectors
        self.use_global_vector = use_global_vector
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_global_self_attn = use_global_self_attn
        self.global_dim_ratio = global_dim_ratio

        if self.use_inter_ffn:
            self.ffn_l = nn.ModuleList(
                [PositionwiseFFN(
                    units=dim,
                    hidden_size=4 * dim,
                    activation_dropout=ffn_drop,
                    dropout=ffn_drop,
                    gated_proj=gated_ffn,
                    activation=activation,
                    normalization=norm_layer,
                    pre_norm=True,
                    linear_init_mode=ffn_linear_init_mode,
                    ffn2_linear_init_mode=ffn2_linear_init_mode,
                    norm_init_mode=norm_init_mode,)
                    for _ in range(self.num_attn)])
            if self.use_global_vector_ffn and self.use_global_vector:
                self.global_ffn_l = nn.ModuleList(
                    [PositionwiseFFN(
                        units=global_dim_ratio * dim,
                        hidden_size=global_dim_ratio * 4 * dim,
                        activation_dropout=ffn_drop,
                        dropout=ffn_drop,
                        gated_proj=gated_ffn,
                        activation=activation,
                        normalization=norm_layer,
                        pre_norm=True,
                        linear_init_mode=ffn_linear_init_mode,
                        ffn2_linear_init_mode=ffn2_linear_init_mode,
                        norm_init_mode=norm_init_mode,)
                        for _ in range(self.num_attn)])
        else:
            self.ffn_l = nn.ModuleList(
                [PositionwiseFFN(
                    units=dim, hidden_size=4 * dim,
                    activation_dropout=ffn_drop,
                    dropout=ffn_drop,
                    gated_proj=gated_ffn, activation=activation,
                    normalization=norm_layer,
                    pre_norm=True,
                    linear_init_mode=ffn_linear_init_mode,
                    ffn2_linear_init_mode=ffn2_linear_init_mode,
                    norm_init_mode=norm_init_mode,)])
            if self.use_global_vector_ffn and self.use_global_vector:
                self.global_ffn_l = nn.ModuleList(
                    [PositionwiseFFN(
                        units=global_dim_ratio * dim,
                        hidden_size=global_dim_ratio * 4 * dim,
                        activation_dropout=ffn_drop,
                        dropout=ffn_drop,
                        gated_proj=gated_ffn, activation=activation,
                        normalization=norm_layer,
                        pre_norm=True,
                        linear_init_mode=ffn_linear_init_mode,
                        ffn2_linear_init_mode=ffn2_linear_init_mode,
                        norm_init_mode=norm_init_mode,)])
        self.attn_l = nn.ModuleList(
            [CuboidSelfAttentionLayer(
                dim=dim, num_heads=num_heads,
                cuboid_size=ele_cuboid_size,
                shift_size=ele_shift_size,
                strategy=ele_strategy,
                padding_type=padding_type,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                use_global_vector=use_global_vector,
                use_global_self_attn=use_global_self_attn,
                separate_global_qkv=separate_global_qkv,
                global_dim_ratio=global_dim_ratio,
                checkpoint_level=checkpoint_level,
                use_relative_pos=use_relative_pos,
                use_final_proj=use_final_proj,
                attn_linear_init_mode=attn_linear_init_mode,
                ffn_linear_init_mode=attn_proj_linear_init_mode,
                norm_init_mode=norm_init_mode,)
                for ele_cuboid_size, ele_shift_size, ele_strategy
                in zip(block_cuboid_size, block_shift_size, block_strategy)])

    def reset_parameters(self):
        for m in self.ffn_l:
            m.reset_parameters()
        if self.use_global_vector_ffn and self.use_global_vector:
            for m in self.global_ffn_l:
                m.reset_parameters()
        for m in self.attn_l:
            m.reset_parameters()

    def forward(self, x, global_vectors=None):
        if self.use_inter_ffn:
            if self.use_global_vector:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    if self.checkpoint_level >= 2 and self.training:
                        x_out, global_vectors_out = checkpoint.checkpoint(attn, x, global_vectors)
                    else:
                        x_out, global_vectors_out = attn(x, global_vectors)
                    x = x + x_out
                    global_vectors = global_vectors + global_vectors_out

                    if self.checkpoint_level >= 1 and self.training:
                        x = checkpoint.checkpoint(ffn, x)
                        if self.use_global_vector_ffn:
                            global_vectors = checkpoint.checkpoint(self.global_ffn_l[idx], global_vectors)
                    else:
                        x = ffn(x)
                        if self.use_global_vector_ffn:
                            global_vectors = self.global_ffn_l[idx](global_vectors)
                return x, global_vectors
            else:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    if self.checkpoint_level >= 2 and self.training:
                        x = x + checkpoint.checkpoint(attn, x)
                    else:
                        x = x + attn(x)
                    if self.checkpoint_level >= 1 and self.training:
                        x = checkpoint.checkpoint(ffn, x)
                    else:
                        x = ffn(x)
                return x
        else:
            if self.use_global_vector:
                for idx, attn in enumerate(self.attn_l):
                    if self.checkpoint_level >= 2 and self.training:
                        x_out, global_vectors_out = checkpoint.checkpoint(attn, x, global_vectors)
                    else:
                        x_out, global_vectors_out = attn(x, global_vectors)
                    x = x + x_out
                    global_vectors = global_vectors + global_vectors_out
                if self.checkpoint_level >= 1 and self.training:
                    x = checkpoint.checkpoint(self.ffn_l[0], x)
                    if self.use_global_vector_ffn:
                        global_vectors = checkpoint.checkpoint(self.global_ffn_l[0], global_vectors)
                else:
                    x = self.ffn_l[0](x)
                    if self.use_global_vector_ffn:
                        global_vectors = self.global_ffn_l[0](global_vectors)
                return x, global_vectors
            else:
                for idx, attn in enumerate(self.attn_l):
                    if self.checkpoint_level >= 2 and self.training:
                        out = checkpoint.checkpoint(attn, x)
                    else:
                        out = attn(x)
                    x = x + out
                if self.checkpoint_level >= 1 and self.training:
                    x = checkpoint.checkpoint(self.ffn_l[0], x)
                else:
                    x = self.ffn_l[0](x)
                return x
