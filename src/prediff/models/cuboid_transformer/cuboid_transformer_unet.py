import torch
from torch import nn
from einops import rearrange

from ..time_embed import TimeEmbedLayer, TimeEmbedResBlock
from .cuboid_transformer import PosEmbed, Upsample3DLayer, PatchMerging3D, StackCuboidSelfAttentionBlock
from .cuboid_transformer_patterns import CuboidSelfAttentionPatterns
from ..utils import timestep_embedding, apply_initialization, round_to


class CuboidTransformerUNet(nn.Module):
    r"""
    U-Net style CuboidTransformer that parameterizes `p(x_{t-1}|x_t)`.
    It takes `x_t`, `t` as input.
    The conditioning can be concatenated to the input like the U-Net in FVD paper.

    For each block, we apply the StackCuboidSelfAttention in U-Net style

        x --> attn --> downscale --> ... --> z --> attn --> upscale --> ... --> out

    Besides, we insert the embeddings of the timesteps `t` before each cuboid attention blocks.
    """
    def __init__(self,
                 input_shape,
                 target_shape,
                 base_units=128,
                 block_units=None,
                 scale_alpha=1.0,
                 depth=[4, 4, 4],
                 downsample=2,
                 downsample_type='patch_merge',
                 upsample_type="upsample",
                 upsample_kernel_size=3,
                 block_attn_patterns=None,
                 block_cuboid_size=[(4, 4, 4),
                                    (4, 4, 4)],
                 block_cuboid_strategy=[('l', 'l', 'l'),
                                        ('d', 'd', 'd')],
                 block_cuboid_shift_size=[(0, 0, 0),
                                          (0, 0, 0)],
                 num_heads=4,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 ffn_drop=0.0,
                 ffn_activation='leaky',
                 gated_ffn=False,
                 norm_layer='layer_norm',
                 use_inter_ffn=True,
                 hierarchical_pos_embed=False,
                 pos_embed_type='t+h+w',
                 padding_type='ignore',
                 checkpoint_level=True,
                 use_relative_pos=True,
                 self_attn_use_final_proj=True,
                 # global vectors
                 num_global_vectors=False,
                 use_global_vector_ffn=True,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 # initialization
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 ffn2_linear_init_mode="2",
                 attn_proj_linear_init_mode="2",
                 conv_init_mode="0",
                 down_linear_init_mode="0",
                 up_linear_init_mode="0",
                 global_proj_linear_init_mode="2",
                 norm_init_mode="0",
                 # timestep embedding for diffusion
                 time_embed_channels_mult=4,
                 time_embed_use_scale_shift_norm=False,
                 time_embed_dropout=0.0,
                 unet_res_connect=True,
                 ):
        super(CuboidTransformerUNet, self).__init__()
        # initialization mode
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.ffn2_linear_init_mode = ffn2_linear_init_mode
        self.attn_proj_linear_init_mode = attn_proj_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_linear_init_mode = down_linear_init_mode
        self.up_linear_init_mode = up_linear_init_mode
        self.global_proj_linear_init_mode = global_proj_linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.input_shape = input_shape
        self.target_shape = target_shape
        self.num_blocks = len(depth)
        self.depth = depth
        self.base_units = base_units
        self.scale_alpha = scale_alpha
        self.downsample = downsample
        self.downsample_type = downsample_type
        self.upsample_type = upsample_type
        self.upsample_kernel_size = upsample_kernel_size
        if not isinstance(downsample, (tuple, list)):
            downsample = (1, downsample, downsample)
        if block_units is None:
            block_units = [round_to(base_units * int((max(downsample) ** scale_alpha) ** i), 4)
                           for i in range(self.num_blocks)]
        else:
            assert len(block_units) == self.num_blocks and block_units[0] == base_units
        self.block_units = block_units
        self.hierarchical_pos_embed = hierarchical_pos_embed
        self.checkpoint_level = checkpoint_level
        self.num_global_vectors = num_global_vectors
        use_global_vector = num_global_vectors > 0
        self.use_global_vector = use_global_vector
        if global_dim_ratio != 1:
            assert separate_global_qkv is True, \
                f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio
        self.use_global_vector_ffn = use_global_vector_ffn

        self.time_embed_channels_mult = time_embed_channels_mult
        self.time_embed_channels = self.block_units[0] * time_embed_channels_mult
        self.time_embed_use_scale_shift_norm = time_embed_use_scale_shift_norm
        self.time_embed_dropout = time_embed_dropout
        self.unet_res_connect = unet_res_connect

        if self.use_global_vector:
            self.init_global_vectors = nn.Parameter(
                torch.zeros((self.num_global_vectors, global_dim_ratio*base_units)))

        T_in, H_in, W_in, C_in = input_shape
        T_out, H_out, W_out, C_out = target_shape
        assert H_in == H_out and W_in == W_out and C_in == C_out
        self.in_len = T_in
        self.out_len = T_out
        self.first_proj = TimeEmbedResBlock(
            channels=self.data_shape[-1],
            emb_channels=None,
            dropout=proj_drop,
            out_channels=self.base_units,
            use_conv=False,
            use_embed=False,
            use_scale_shift_norm=False,
            dims=3,
            use_checkpoint=False,
            up=False,
            down=False, )
        self.pos_embed = PosEmbed(
            embed_dim=base_units, typ=pos_embed_type,
            maxT=self.data_shape[0], maxH=H_in, maxW=W_in)

        # diffusion time embed
        self.time_embed = TimeEmbedLayer(base_channels=self.block_units[0],
                                         time_embed_channels=self.time_embed_channels)
        # # inner U-Net
        if self.num_blocks > 1:
            # Construct downsampling layers
            if downsample_type == 'patch_merge':
                self.downsample_layers = nn.ModuleList(
                    [PatchMerging3D(dim=self.block_units[i],
                                    downsample=downsample,
                                    # downsample=(1, 1, 1),
                                    padding_type=padding_type,
                                    out_dim=self.block_units[i + 1],
                                    linear_init_mode=down_linear_init_mode,
                                    norm_init_mode=norm_init_mode)
                     for i in range(self.num_blocks - 1)])
            else:
                raise NotImplementedError
            if self.use_global_vector:
                self.down_layer_global_proj = nn.ModuleList(
                    [nn.Linear(in_features=global_dim_ratio*self.block_units[i],
                               out_features=global_dim_ratio*self.block_units[i + 1])
                     for i in range(self.num_blocks - 1)])
            # Construct upsampling layers
            if self.upsample_type == "upsample":
                self.upsample_layers = nn.ModuleList([
                    Upsample3DLayer(
                        dim=self.mem_shapes[i + 1][-1],
                        out_dim=self.mem_shapes[i][-1],
                        target_size=self.mem_shapes[i][:3],
                        kernel_size=upsample_kernel_size,
                        temporal_upsample=False,
                        conv_init_mode=conv_init_mode,
                    )
                    for i in range(self.num_blocks - 1)])
            else:
                raise NotImplementedError
            if self.use_global_vector:
                self.up_layer_global_proj = nn.ModuleList(
                    [nn.Linear(in_features=global_dim_ratio*self.block_units[i + 1],
                               out_features=global_dim_ratio*self.block_units[i])
                     for i in range(self.num_blocks - 1)])
            if self.hierarchical_pos_embed:
                self.down_hierarchical_pos_embed_l = nn.ModuleList([
                    PosEmbed(embed_dim=self.block_units[i], typ=pos_embed_type,
                             maxT=self.mem_shapes[i][0], maxH=self.mem_shapes[i][1], maxW=self.mem_shapes[i][2])
                    for i in range(self.num_blocks - 1)])
                self.up_hierarchical_pos_embed_l = nn.ModuleList([
                    PosEmbed(embed_dim=self.block_units[i], typ=pos_embed_type,
                             maxT=self.mem_shapes[i][0], maxH=self.mem_shapes[i][1], maxW=self.mem_shapes[i][2])
                    for i in range(self.num_blocks - 1)])

        if block_attn_patterns is not None:
            if isinstance(block_attn_patterns, (tuple, list)):
                assert len(block_attn_patterns) == self.num_blocks
            else:
                block_attn_patterns = [block_attn_patterns for _ in range(self.num_blocks)]
            block_cuboid_size = []
            block_cuboid_strategy = []
            block_cuboid_shift_size = []
            for idx, key in enumerate(block_attn_patterns):
                func = CuboidSelfAttentionPatterns.get(key)
                cuboid_size, strategy, shift_size = func(self.mem_shapes[idx])
                block_cuboid_size.append(cuboid_size)
                block_cuboid_strategy.append(strategy)
                block_cuboid_shift_size.append(shift_size)
        else:
            if not isinstance(block_cuboid_size[0][0], (list, tuple)):
                block_cuboid_size = [block_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert len(block_cuboid_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_cuboid_size={block_cuboid_size}'

            if not isinstance(block_cuboid_strategy[0][0], (list, tuple)):
                block_cuboid_strategy = [block_cuboid_strategy for _ in range(self.num_blocks)]
            else:
                assert len(block_cuboid_strategy) == self.num_blocks,\
                    f'Incorrect input format! Received block_strategy={block_cuboid_strategy}'

            if not isinstance(block_cuboid_shift_size[0][0], (list, tuple)):
                block_cuboid_shift_size = [block_cuboid_shift_size for _ in range(self.num_blocks)]
            else:
                assert len(block_cuboid_shift_size) == self.num_blocks,\
                    f'Incorrect input format! Received block_shift_size={block_cuboid_shift_size}'
        self.block_cuboid_size = block_cuboid_size
        self.block_cuboid_strategy = block_cuboid_strategy
        self.block_cuboid_shift_size = block_cuboid_shift_size

        # cuboid self attention blocks
        down_self_blocks = []
        up_self_blocks = []
        # ResBlocks that incorporate `time_embed`
        down_time_embed_blocks = []
        up_time_embed_blocks = []
        for i in range(self.num_blocks):
            down_time_embed_blocks.append(
                TimeEmbedResBlock(
                    channels=self.mem_shapes[i][-1],
                    emb_channels=self.time_embed_channels,
                    dropout=self.time_embed_dropout,
                    out_channels=self.mem_shapes[i][-1],
                    use_conv=False,
                    use_embed=True,
                    use_scale_shift_norm=self.time_embed_use_scale_shift_norm,
                    dims=3,
                    use_checkpoint=checkpoint_level >= 1,
                    up=False,
                    down=False,))

            ele_depth = depth[i]
            stack_cuboid_blocks =\
                [StackCuboidSelfAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_size=block_cuboid_size[i],
                    block_strategy=block_cuboid_strategy[i],
                    block_shift_size=block_cuboid_shift_size[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    activation=ffn_activation,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    padding_type=padding_type,
                    use_global_vector=use_global_vector,
                    use_global_vector_ffn=use_global_vector_ffn,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=self_attn_use_final_proj,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    ffn2_linear_init_mode=ffn2_linear_init_mode,
                    attn_proj_linear_init_mode=attn_proj_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                ) for _ in range(ele_depth)]
            down_self_blocks.append(nn.ModuleList(stack_cuboid_blocks))

            up_time_embed_blocks.append(
                TimeEmbedResBlock(
                    channels=self.mem_shapes[i][-1],
                    emb_channels=self.time_embed_channels,
                    dropout=self.time_embed_dropout,
                    out_channels=self.mem_shapes[i][-1],
                    use_conv=False,
                    use_embed=True,
                    use_scale_shift_norm=self.time_embed_use_scale_shift_norm,
                    dims=3,
                    use_checkpoint=checkpoint_level >= 1,
                    up=False,
                    down=False,))

            stack_cuboid_blocks = \
                [StackCuboidSelfAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_size=block_cuboid_size[i],
                    block_strategy=block_cuboid_strategy[i],
                    block_shift_size=block_cuboid_shift_size[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    activation=ffn_activation,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    padding_type=padding_type,
                    use_global_vector=use_global_vector,
                    use_global_vector_ffn=use_global_vector_ffn,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=self_attn_use_final_proj,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    ffn2_linear_init_mode=ffn2_linear_init_mode,
                    attn_proj_linear_init_mode=attn_proj_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                ) for _ in range(ele_depth)]
            up_self_blocks.append(nn.ModuleList(stack_cuboid_blocks))
        self.down_self_blocks = nn.ModuleList(down_self_blocks)
        self.up_self_blocks = nn.ModuleList(up_self_blocks)
        self.down_time_embed_blocks = nn.ModuleList(down_time_embed_blocks)
        self.up_time_embed_blocks = nn.ModuleList(up_time_embed_blocks)
        self.final_proj = nn.Linear(self.base_units, C_out)

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_global_vectors > 0:
            nn.init.trunc_normal_(self.init_global_vectors, std=.02)
        self.first_proj.reset_parameters()
        apply_initialization(self.final_proj, linear_mode="2")
        self.pos_embed.reset_parameters()
        # inner U-Net
        for ms in self.down_self_blocks:
            for m in ms:
                m.reset_parameters()
        for m in self.down_time_embed_blocks:
            m.reset_parameters()
        for ms in self.up_self_blocks:
            for m in ms:
                m.reset_parameters()
        for m in self.up_time_embed_blocks:
            m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.downsample_layers:
                m.reset_parameters()
            for m in self.upsample_layers:
                m.reset_parameters()
            if self.use_global_vector:
                apply_initialization(self.down_layer_global_proj,
                                     linear_mode=self.global_proj_linear_init_mode)
                apply_initialization(self.up_layer_global_proj,
                                     linear_mode=self.global_proj_linear_init_mode)
        if self.hierarchical_pos_embed:
            for m in self.down_hierarchical_pos_embed_l:
                m.reset_parameters()
            for m in self.up_hierarchical_pos_embed_l:
                m.reset_parameters()

    @property
    def data_shape(self):
        if not hasattr(self, "_data_shape"):
            T_in, H_in, W_in, C_in = self.input_shape
            T_out, H_out, W_out, C_out = self.target_shape
            assert H_in == H_out and W_in == W_out and C_in == C_out
            self._data_shape = (T_in + T_out, H_in, W_in, C_in + 1)  # concat mask to indicate observation and target
        return self._data_shape

    @property
    def mem_shapes(self):
        """Get the shape of the output memory based on the input shape. This can be used for constructing the decoder.

        Returns
        -------
        mem_shapes
            A list of shapes of the output memory
        """
        inner_data_shape = tuple(self.data_shape)[:3] + (self.base_units, )
        if self.num_blocks == 1:
            return [inner_data_shape]
        else:
            mem_shapes = [inner_data_shape]
            curr_shape = inner_data_shape
            for down_layer in self.downsample_layers:
                curr_shape = down_layer.get_out_shape(curr_shape)
                mem_shapes.append(curr_shape)
            return mem_shapes

    def forward(self, x, t, cond, verbose=False):
        """

        Parameters
        ----------
        x:  torch.Tensor
            Shape (B, T_out, H, W, C)
        t:  torch.Tensor
            Shape (B, )
        cond:   torch.Tensor
            Shape (B, T_in, H, W, C)
        verbose:    bool

        Returns
        -------
        out:    torch.Tensor
            Shape (B, T, H, W, C)
        """
        batch_size = x.shape[0]
        x = torch.cat([cond, x], dim=1)
        obs_indicator = torch.ones_like(x[..., :1])
        obs_indicator[:, self.in_len:, ...] = 0.0
        x = torch.cat([x, obs_indicator], dim=-1)
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.first_proj(x)
        x = rearrange(x, "b c t h w -> b t h w c")
        if self.use_global_vector:
            global_vectors = self.init_global_vectors \
                .expand(batch_size, self.num_global_vectors, self.global_dim_ratio * self.base_units)
        x = self.pos_embed(x)
        # inner U-Net
        t_emb = self.time_embed(timestep_embedding(t, self.block_units[0]))
        if self.unet_res_connect:
            res_connect_l = []
        if verbose:
            print("downsampling")
        for i in range(self.num_blocks):
            # Downample
            if i > 0:
                x = self.downsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.down_hierarchical_pos_embed_l[i - 1](x)
                if self.use_global_vector:
                    global_vectors = self.down_layer_global_proj[i - 1](global_vectors)
            for idx in range(self.depth[i]):
                x = rearrange(x, "b t h w c -> b c t h w")
                x = self.down_time_embed_blocks[i](x, t_emb)
                x = rearrange(x, "b c t h w -> b t h w c")
                if self.use_global_vector:
                    print(f"x.shape = {x.shape}")
                    print(f"global_vectors.shape = {global_vectors.shape}")
                    x, global_vectors = self.down_self_blocks[i][idx](x, global_vectors)
                else:
                    x = self.down_self_blocks[i][idx](x)
            if verbose:
                print(f"x.shape = {x.shape}")
                if global_vectors is not None:
                    print(f"global_vectors.shape = {global_vectors.shape}")
            if self.unet_res_connect and i < self.num_blocks - 1:
                res_connect_l.append(x)
        if verbose:
            print("upsampling")
        for i in range(self.num_blocks - 1, -1, -1):
            if verbose:
                print(f"x.shape = {x.shape}")
                if global_vectors is not None:
                    print(f"global_vectors.shape = {global_vectors.shape}")
            if self.unet_res_connect and i < self.num_blocks - 1:
                x = x + res_connect_l[i]
            for idx in range(self.depth[i]):
                x = rearrange(x, "b t h w c -> b c t h w")
                x = self.up_time_embed_blocks[i](x, t_emb)
                x = rearrange(x, "b c t h w -> b t h w c")
                if self.use_global_vector:
                    x, global_vectors = self.up_self_blocks[i][idx](x, global_vectors)
                    # TODO: the last global_vectors proj layer can not get gradients
                    #  since the updated global_vectors are not used.
                else:
                    x = self.up_self_blocks[i][idx](x)
            # Upsample
            if i > 0:
                x = self.upsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.up_hierarchical_pos_embed_l[i - 1](x)
                if self.use_global_vector:
                    global_vectors = self.up_layer_global_proj[i - 1](global_vectors)
        x = self.final_proj(x[:, self.in_len:, ...])
        return x
