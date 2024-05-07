import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from .bcam import BCAM, GACD, EFN


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # cat op
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN feed-forward network
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()  # B, 3, H, W
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class MultiModalSwinTransformer(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.0,
                 args=None
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        if args.ytvos_2d_swin_3d_pwam:
            print("Using 2D Swin Transformer as backbone with 3D PWAM!")
        elif args.ytvos_2d_swin_pwam:
            print("Using 2D Swin Transformer as backbone with PWAM!")

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if args.ytvos_2d_swin_3d_pwam:
                layer = MMBasicLayer_2d_swin_3d_pwam(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    num_heads_fusion=num_heads_fusion[i_layer],
                    fusion_drop=fusion_drop,
                    args=args
                )
            else:
                layer = MMBasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    num_heads_fusion=num_heads_fusion[i_layer],
                    fusion_drop=fusion_drop,
                    args=args
                )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=('upernet' in pretrained), logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, l, l_mask):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, l, l_mask)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)  # output of a Block has shape (B, H*W, dim)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MultiModalSwinTransformer, self).train(mode)
        self._freeze_stages()


class MMBasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 num_heads_fusion=1,
                 fusion_drop=0.0,
                 args=None
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.version = args.version
        self.fuse = args.fuse
        self.hs = args.hs
        self.lazy_pred = args.lazy_pred
        lg_act_layer_dict = {'tanh': nn.Tanh,
                             'sigmoid': nn.Sigmoid}
        lg_act_layer = lg_act_layer_dict[args.lg_act_layer]

        # build blocks 原本的2dSwin
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # fuse before downsampling
        if args.bcam:
            self.fusion = BCAM(dim,  # both visual input and for combining, num of channels
                               dim,  # v_in
                               768   #768,  # l_in_channs
                               )
        elif args.gacd:
            self.fusion = GACD(dim,  # both visual input and for combining, num of channels
                               dim,  # v_in
                               768,  #768,  # l_in_channs
                               num_heads=num_heads_fusion)
        elif args.efn:
            print('Initializing EFN attention!')
            self.fusion = EFN(dim,  # both visual input and for combining, num of channels
                              dim,  # v_in
                              768,  # l_in
                             )
        else:
            self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                               dim,  # v_in
                               768,  # l_in
                               dim,  # key
                               dim,  # value
                               num_heads=num_heads_fusion,
                               dropout=fusion_drop,
                               attention=(args.fuse != 'simple'),
                               att_norm_layer_type=args.att_norm_layer_type
                               )

        if self.version != 'default':
            pass
        else:
            self.res_gate = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.ReLU(),
                nn.Linear(dim, dim, bias=False),
                lg_act_layer()
                # nn.Tanh()
                # nn.Sigmoid()
            )
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        
        if self.version != 'default':
            return
        # initialize the gate to 0
        nn.init.zeros_(self.res_gate[0].weight)
        nn.init.zeros_(self.res_gate[2].weight)

    def forward(self, x, H, W, l, l_mask):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # Swin blocks
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # output of a Block has shape (B, H*W, dim)

        if self.lazy_pred:
            x_out = x

        # PWAM fusion
        x_residual = self.fusion(x, l, l_mask) # PWAM输出
        if self.version == "default":
            # apply a gate on the residual
            x = x + (self.res_gate(x_residual) * x_residual)
        elif self.version == "no_gate":
            x = x + x_residual

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            if self.hs:
                return x, H, W, x_down, Wh, Ww
            if self.lazy_pred:
                return x_out, H, W, x_down, Wh, Ww
            return x_residual, H, W, x_down, Wh, Ww
        else:
            if self.hs:
                return x, H, W, x, H, W
            if self.lazy_pred:
                return x_out, H, W, x, H, W
            return x_residual, H, W, x, H, W
        

def _3d_kernel_size_to_padding_size(kernel_size):
    ## this function currently assumes that the kernel_size is either (3, 3, 3) or (3, 1, 1)
    ## and is used when stride is 1
    if kernel_size == (3, 3, 3):
        return 1, 1, 1

    elif kernel_size == (1, 1, 1):
        return 0, 0, 0

    elif kernel_size == (3, 1, 1):
        return 1, 0, 0

    elif kernel_size == (1, 3, 3):
        return 0, 1, 1

    else:
        return None


class SepTPWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 conv3d_kernel_size_t=(3, 1, 1), conv3d_kernel_size_s=(1, 1, 1), w_3x3=False, mm_3x3=False,
                 w_3=False, mm_3=False, sum_3_kernel_size=None, cat_reduce_kernel_size=None,
                 w_t3x3_s1x1=None, mm_t3x3_s1x1=None, args=None
                 ):
        super(SepTPWAM, self).__init__()
        # input x shape: (B, D, H, W, C)
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        # conv3d_kernel_size  # (depth, height, width)

        self.num_heads = num_heads
        padding_t = _3d_kernel_size_to_padding_size(conv3d_kernel_size_t)  # 3 -> 1; 1 -> 0
        padding_s = _3d_kernel_size_to_padding_size(conv3d_kernel_size_s)  # 3 -> 1; 1 -> 0
        self.sum_3_kernel_size = sum_3_kernel_size
        self.cat_reduce_kernel_size = cat_reduce_kernel_size
        padding_sum = _3d_kernel_size_to_padding_size(sum_3_kernel_size)  # 3 -> 1; 1 -> 0
        padding_cat = _3d_kernel_size_to_padding_size(cat_reduce_kernel_size)  # 3 -> 1; 1 -> 0
        out_channels = value_channels
        self.w_3x3 = w_3x3
        self.w_3 = w_3
        self.w_t3x3_s1x1 = w_t3x3_s1x1
        self.mm_3x3 = mm_3x3
        self.mm_3 = mm_3
        self.mm_t3x3_s1x1 = mm_t3x3_s1x1
        self.s_tanh_plus_1_gate_1_q = args.s_tanh_plus_1_gate_1_q
        self.s_tanh_plus_1_gate_1_v = args.s_tanh_plus_1_gate_1_v
        self.t_tanh_plus_1_gate_1_q = args.t_tanh_plus_1_gate_1_q
        self.t_tanh_plus_1_gate_1_v = args.t_tanh_plus_1_gate_1_v

        # The outer layer; top red square
        self.temporal_vis_project = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=conv3d_kernel_size_t,
                                                              stride=1, padding=padding_t),
                                                  nn.GELU(),
                                                  nn.Dropout(dropout)
                                                  )

        self.spatial_vis_project = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=conv3d_kernel_size_s,
                                                              stride=1, padding=padding_s),
                                                 nn.GELU(),
                                                 nn.Dropout(dropout)
                                                 )
        if self.s_tanh_plus_1_gate_1_v:
            self.s_gate_v = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.ReLU(),
                                          nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.Tanh()
                                          #    nn.Sigmoid()
                                          )

        if self.t_tanh_plus_1_gate_1_v:
            self.t_gate_v = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.ReLU(),
                                          nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.Tanh()
                                          #    nn.Sigmoid()
                                          )

        if self.sum_3_kernel_size is not None:
            self.vis_fuse = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=self.sum_3_kernel_size,
                                                    stride=1, padding=padding_sum),
                                          nn.GELU(),
                                          nn.Dropout(dropout)
                                          )
        elif self.cat_reduce_kernel_size is not None:
            self.vis_fuse = nn.Sequential(nn.Conv3d(2*dim, dim, kernel_size=self.cat_reduce_kernel_size,
                                                    stride=1, padding=padding_cat),
                                          nn.GELU(),
                                          nn.Dropout(dropout)
                                          )

        # Queries: visual features t for temporal s for spatial, sum the two
        self.f_query_t = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_t, stride=1, padding=padding_t),
            nn.InstanceNorm3d(key_channels),
        )

        self.f_query_s = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_s, stride=1, padding=padding_s),
            nn.InstanceNorm3d(key_channels),
        )

        if self.s_tanh_plus_1_gate_1_q:
            self.s_gate_q = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.ReLU(),
                                          nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.Tanh()
                                          #    nn.Sigmoid()
                                          )
        if self.t_tanh_plus_1_gate_1_q:
            self.t_gate_q = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.ReLU(),
                                          nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                                          nn.Tanh()
                                          #    nn.Sigmoid()
                                          )

        if self.sum_3_kernel_size is not None:
            self.f_fuse = nn.Sequential(
                nn.Conv3d(key_channels, key_channels, kernel_size=self.sum_3_kernel_size,
                          stride=1, padding=padding_sum),
                nn.InstanceNorm3d(key_channels),
            )
        elif self.cat_reduce_kernel_size is not None:
            self.f_fuse = nn.Sequential(
                nn.Conv3d(2*key_channels, key_channels, kernel_size=self.cat_reduce_kernel_size,
                          stride=1, padding=padding_cat),
                nn.InstanceNorm3d(key_channels),
            )
        # the above

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(l_in_channels, key_channels, kernel_size=1, stride=1),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(l_in_channels, value_channels, kernel_size=1, stride=1),
        )

        if self.w_3x3:
            self.W = nn.Sequential(
                nn.Conv3d(value_channels, out_channels, kernel_size=conv3d_kernel_size_t, stride=1, padding=padding_t),
                nn.InstanceNorm3d(out_channels),
            )
        elif self.w_3:
            self.W = nn.Sequential(
                nn.Conv3d(value_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                nn.InstanceNorm3d(out_channels),
            )
        elif self.w_t3x3_s1x1:
            self.W_t = nn.Sequential(
                nn.Conv3d(value_channels, out_channels, kernel_size=conv3d_kernel_size_t, stride=1, padding=padding_t),
                nn.InstanceNorm3d(out_channels),
            )
            self.W_s = nn.Sequential(
                nn.Conv3d(value_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm3d(out_channels),
            )
        else:
            # wrap-up projection
            self.W = nn.Sequential(
                nn.Conv1d(value_channels, out_channels, kernel_size=1, stride=1),
                nn.InstanceNorm1d(out_channels),
            )

        if self.mm_3x3:
            self.project_mm = nn.Sequential(nn.Conv3d(value_channels, value_channels, kernel_size=conv3d_kernel_size_t,
                                                      stride=1, padding=padding_t),
                                            nn.GELU(),
                                            nn.Dropout(dropout)
                                            )
        elif self.mm_3:
            self.project_mm = nn.Sequential(nn.Conv3d(value_channels, value_channels, kernel_size=(1, 3, 3),
                                                      stride=1, padding=(0, 1, 1)),
                                            nn.GELU(),
                                            nn.Dropout(dropout)
                                            )
        elif self.mm_t3x3_s1x1:
            self.project_mm_t = nn.Sequential(nn.Conv3d(value_channels, value_channels, kernel_size=conv3d_kernel_size_t,
                                                        stride=1, padding=padding_t),
                                              nn.GELU(),
                                              nn.Dropout(dropout)
                                              )
            self.project_mm_s = nn.Sequential(nn.Conv3d(value_channels, value_channels, kernel_size=1,
                                                        stride=1, padding=0),
                                              nn.GELU(),
                                              nn.Dropout(dropout)
                                              )

        else:
            self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                            nn.GELU(),
                                            nn.Dropout(dropout)
                                            )

    def forward(self, x, l, l_mask):
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, D, H, W, C = x.size()
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        temporal_vis = self.temporal_vis_project(x)  # (B, dim, T, H, W) because Conv3D
        # temporal_vis = temporal_vis.view(B, C, D * H * W)  # (B, dim, THW) to make shapes consistent

        spatial_vis = self.spatial_vis_project(x)  # (B, dim, T, H, W) because Conv3D
        # spatial_vis = spatial_vis.view(B, C, D * H * W)  # (B, dim, THW) to make shapes consistent

        if self.t_tanh_plus_1_gate_1_v:
            temporal_vis = temporal_vis + self.t_gate_v(temporal_vis) * temporal_vis  # (1 + tanh(temporal_vis)) * temporal_vis
        if self.s_tanh_plus_1_gate_1_v:
            spatial_vis = spatial_vis + self.s_gate_v(spatial_vis) * spatial_vis  # (1 + tanh(spatial_vis)) * spatial_vis

        if self.cat_reduce_kernel_size is not None:
            ts_vis = torch.cat([temporal_vis, spatial_vis], dim=1)  # (B, 2*dim, T, H, W)
            ts_vis = self.vis_fuse(ts_vis)  # (B, dim, T, H, W) because Conv3D
        else:
            ts_vis = temporal_vis + spatial_vis  # sum to fuse; (B, dim, T, H, W)
            if self.sum_3_kernel_size is not None:
                ts_vis = self.vis_fuse(ts_vis)  # (B, dim, T, H, W) because Conv3D

        ts_vis = ts_vis.view(B, C, D * H * W)  # (B, dim, THW)

        #### temporal spatial image language attention forward transferred here
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        query_t = self.f_query_t(x)  # (B, key_channels, T, H, W) because Conv3D
        query_s = self.f_query_s(x)  # (B, key_channels, T, H, W) because Conv3D

        if self.t_tanh_plus_1_gate_1_q:
            query_t = query_t + self.t_gate_q(query_t) * query_t  # (1 + tanh(query_t)) * query_t
        if self.s_tanh_plus_1_gate_1_q:
            query_s = query_s + self.s_gate_q(query_s) * query_s  # (1 + tanh(query_s)) * query_s

        if self.cat_reduce_kernel_size is not None:
            query = torch.cat([query_t, query_s], dim=1)  # (B, 2*dim, T, H, W)
            query = self.f_fuse(query)  # (B, dim, T, H, W) because Conv3D
        else:
            query = query_t + query_s  # (B, key_channels, T, H, W)
            if self.sum_3_kernel_size is not None:
                query = self.f_fuse(query)  # (B, key_channels, T, H, W) because Conv3D

        # getting ready for attentional steps #
        query = query.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        query = query.view(B, D*H*W, C)  # (B, DHW, C)

        key = self.f_key(l)  # (B, C, N_l)
        value = self.f_value(l)  # (B, C, N_l)

        key = key * l_mask  # (B, C, N_l)
        value = value * l_mask  # (B, C, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, D*H*W, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, T*H*W, C//self.num_heads)
        key = key.reshape(B, self.num_heads, C//self.num_heads, n_l)
        # (b, num_heads, C//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, C//self.num_heads, n_l)
        # (b, num_heads, C//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, T*H*W, N_l)
        sim_map = (C ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, t*h*w, N_l)
        ts_lang = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, T*H*W, C//num_heads)
        ts_lang = ts_lang.permute(0, 2, 1, 3).contiguous().reshape(B, D*H*W, C)  # (B, T*H*W, C)
        ts_lang = ts_lang.permute(0, 2, 1)  # (B, C, THW)
        if self.w_3x3 or self.w_3:
            ts_lang = ts_lang.view(B, C, D, H, W)  # (B, C, T, H, W)
            ts_lang = self.W(ts_lang)  # (B, C, T, H, W)
            ts_lang = ts_lang.view(B, C, D * H * W)  # (B, C, T*H*W)
        elif self.w_t3x3_s1x1:
            ts_lang = ts_lang.view(B, C, D, H, W)  # (B, C, T, H, W)
            ts_lang_t = self.W_t(ts_lang)  # (B, C, T, H, W)
            ts_lang_s = self.W_s(ts_lang)  # (B, C, T, H, W)
            ts_lang = ts_lang_t + ts_lang_s
            ts_lang = ts_lang.view(B, C, D * H * W)  # (B, C, T*H*W)

        else:
            ts_lang = self.W(ts_lang)  # (B, C, THW)
        # the above

        mm = torch.mul(ts_vis, ts_lang)  # (B, dim, T*H*W)
        if self.mm_3x3 or self.mm_3:
            mm = mm.view(B, C, D, H, W)  # (B, C, T, H, W)
            mm = self.project_mm(mm)  # (B, dim, T, H, W)
            mm = mm.view(B, C, D * H * W)  # (B, C, T*H*W)
        elif self.mm_t3x3_s1x1:
            mm = mm.view(B, C, D, H, W)  # (B, C, T, H, W)
            mm_t = self.project_mm_t(mm)  # (B, dim, T, H, W)
            mm_s = self.project_mm_s(mm)  # (B, dim, T, H, W)
            mm = mm_t + mm_s
            mm = mm.view(B, C, D * H * W)  # (B, C, T*H*W)
        else:
            mm = self.project_mm(mm)  # (B, dim, T*H*W)

        mm = mm.permute(0, 2, 1)  # (B, T*H*W, dim)
        return mm


class MMBasicLayer_2d_swin_3d_pwam(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 num_heads_fusion=1,
                 fusion_drop=0.0,
                 args=None
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.version = args.version
        self.fuse = args.fuse
        self.hs = args.hs
        self.lazy_pred = args.lazy_pred
        lg_act_layer_dict = {'tanh': nn.Tanh,
                             'sigmoid': nn.Sigmoid}
        lg_act_layer = lg_act_layer_dict[args.lg_act_layer]

        # For 3d pwam
        self.args = args
        self.is_last_layer = (num_heads == 24) or (num_heads == 32)  # because t & s are 24; b is 32; l is ?
        self.ts_pwam = args.ts_pwam
        self.t_pwam = args.t_pwam  # TPWAM
        self.t_pwam_comp = args.t_pwam_comp  # TPWAMComp
        self.sep_t_pwam = args.sep_t_pwam  # SepTPWAM
        self.seq_t_pwam = args.seq_t_pwam  # SeqTPWAM
        self.sep_t_pwam_inner = args.sep_t_pwam_inner  # inner query only SepTPWAM
        self.sep_seq_t_pwam = args.sep_seq_t_pwam  # SepSeqTPWAM
        self.sep_seq_t_pwam_inner = args.sep_seq_t_pwam_inner  # SepSeqTPWAMInner

        if self.ts_pwam or self.t_pwam or self.t_pwam_comp:
            conv3d_kernel_size = args.conv3d_kernel_size.split('-')  # parse into ['a', 'b', 'c']
            conv3d_kernel_size = tuple([int(a) for a in conv3d_kernel_size])  # e.g., (3, 1, 1)
        elif self.sep_t_pwam or self.seq_t_pwam or self.sep_t_pwam_inner or self.sep_seq_t_pwam \
                or self.sep_seq_t_pwam_inner:
            conv3d_kernel_size_t = args.conv3d_kernel_size_t.split('-')  # parse into ['a', 'b', 'c']
            conv3d_kernel_size_t = tuple([int(a) for a in conv3d_kernel_size_t])  # e.g., (3, 1, 1)

            conv3d_kernel_size_s = args.conv3d_kernel_size_s.split('-')  # parse into ['a', 'b', 'c']
            conv3d_kernel_size_s = tuple([int(a) for a in conv3d_kernel_size_s])  # e.g., (1, 1, 1)

            if self.sep_seq_t_pwam or self.sep_seq_t_pwam_inner:
                conv3d_kernel_size_sq = args.conv3d_kernel_size_sq.split('-')  # parse into ['a', 'b', 'c']
                conv3d_kernel_size_sq = tuple([int(a) for a in conv3d_kernel_size_sq])  # e.g., (1, 1, 1)

            if args.sept_sum_3_kernel_size:
                sept_sum_3_kernel_size = args.sept_sum_3_kernel_size.split('-')  # parse into ['a', 'b', 'c']
                sept_sum_3_kernel_size = tuple([int(a) for a in sept_sum_3_kernel_size])  # e.g., (3, 1, 1)
            else:
                sept_sum_3_kernel_size = None

            if args.sept_cat_reduce_kernel_size:
                sept_cat_reduce_kernel_size = args.sept_cat_reduce_kernel_size.split('-')  # parse into ['a', 'b', 'c']
                sept_cat_reduce_kernel_size = tuple([int(a) for a in sept_cat_reduce_kernel_size])  # e.g., (3, 1, 1)
            else:
                sept_cat_reduce_kernel_size = None

        # build blocks 原本的2dSwin
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # fuse before downsampling
        if args.bcam:
            self.fusion = BCAM(dim,  # both visual input and for combining, num of channels
                               dim,  # v_in
                               768   #768,  # l_in_channs
                               )
        elif args.gacd:
            self.fusion = GACD(dim,  # both visual input and for combining, num of channels
                               dim,  # v_in
                               768,  #768,  # l_in_channs
                               num_heads=num_heads_fusion)
        elif args.efn:
            print('Initializing EFN attention!')
            self.fusion = EFN(dim,  # both visual input and for combining, num of channels
                              dim,  # v_in
                              768,  # l_in
                             )
        # 这里换成3D PWAM @@@
        elif self.sep_t_pwam: 
            self.fusion = SepTPWAM(dim, dim, 768, dim, dim, num_heads=num_heads_fusion, dropout=fusion_drop,
                                   conv3d_kernel_size_t=conv3d_kernel_size_t, conv3d_kernel_size_s=conv3d_kernel_size_s,
                                   w_3x3=args.w_3x3, mm_3x3=args.mm_3x3, w_3=args.w_3, mm_3=args.mm_3,
                                   sum_3_kernel_size=sept_sum_3_kernel_size,
                                   cat_reduce_kernel_size=sept_cat_reduce_kernel_size,
                                   w_t3x3_s1x1=args.w_t3x3_s1x1, mm_t3x3_s1x1=args.mm_t3x3_s1x1,
                                   args=args
                                   )
        else:
            self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                               dim,  # v_in
                               768,  # l_in
                               dim,  # key
                               dim,  # value
                               num_heads=num_heads_fusion,
                               dropout=fusion_drop,
                               attention=(args.fuse != 'simple'),
                               att_norm_layer_type=args.att_norm_layer_type
                               )

        if self.version != 'default':
            pass
        else:
            self.res_gate = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.ReLU(),
                nn.Linear(dim, dim, bias=False),
                lg_act_layer()
                # nn.Tanh()
                # nn.Sigmoid()
            )
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        
        if self.version != 'default':
            return
        # initialize the gate to 0
        nn.init.zeros_(self.res_gate[0].weight)
        nn.init.zeros_(self.res_gate[2].weight)

    def forward(self, x, H, W, l, l_mask):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # Swin blocks
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # output of a Block has shape (B, H*W, dim)

        if self.lazy_pred:
            x_out = x

        # PWAM fusion

        # ours (B*T, H*W, dim)
        # l input shape: (B, l_in_channels, N_l) -> B*T
        # l_mask shape: (B, N_l, 1) -> B*T

        # pwam x input & output both is (B, H*W, dim)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)

        # 3d-pwam
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        # (B, T*H*W, dim)

        # x_residual = self.fusion(x, l, l_mask) # PWAM输出
        # B = self.args.batch_size  # 1
        # D = self.args.num_frames  # 8

        B = l.shape[0]
        D = x.shape[0] // B
        _, _, C = x.shape

        # x [1, 8, 20, 20, 128]
        
        x_reshape = x
        x_residual = self.fusion(x_reshape.view(B, D, H, W, C), l, l_mask)  # the output has shape (B, DHW, C)
        x_residual = x_residual.contiguous().view(B*D, H*W, C)
        
        if self.version == "default":
            # apply a gate on the residual
            x = x + (self.res_gate(x_residual) * x_residual)
        elif self.version == "no_gate":
            x = x + x_residual

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            if self.hs:
                return x, H, W, x_down, Wh, Ww
            if self.lazy_pred:
                return x_out, H, W, x_down, Wh, Ww
            return x_residual, H, W, x_down, Wh, Ww
        else:
            if self.hs:
                return x, H, W, x, H, W
            if self.lazy_pred:
                return x_out, H, W, x, H, W
            return x_residual, H, W, x, H, W


class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 attention=True,  att_norm_layer_type='IN'):
        super(PWAM, self).__init__()
        self.attention = attention
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )
        if attention:
            self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                                l_in_channels,  # l_in
                                                                key_channels,  # key
                                                                value_channels,  # value
                                                                out_channels=value_channels,  # out
                                                                num_heads=num_heads,
                                                                att_norm_layer_type=att_norm_layer_type
                                                                )
        else:
            self.image_lang_att = LangProject(l_in_channels=l_in_channels, l_out_channels=value_channels)
        
        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim) or (B, 1, dim) if ablation

        lang = lang.permute(0, 2, 1)  # (B, dim, H*W) or (B, dim, 1) if ablation

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1,
                 att_norm_layer_type='IN'):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels
        self.att_norm_layer_type = att_norm_layer_type
        att_norm_layer_dict = {'IN': nn.InstanceNorm1d,
                               'BN': nn.BatchNorm1d,
                               'LN': nn.LayerNorm,
                               'none': nn.Identity
                               }
        att_norm_layer = att_norm_layer_dict[att_norm_layer_type]

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            # nn.InstanceNorm1d(self.key_channels),
            att_norm_layer(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            # nn.InstanceNorm1d(self.out_channels),
            att_norm_layer(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        if self.att_norm_layer_type != 'LN':
            query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
            query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        else:
            query = self.f_query[0](x)  # (B, key_channels, H*W) if Conv1D
            query = self.f_query[1](query.permute(0, 2, 1))  # (B, H*W, key_channels) and is output of layer norm

        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        if self.att_norm_layer_type != 'LN':
            out = self.W(out)  # (B, value_channels, HW)
            out = out.permute(0, 2, 1)  # (B, HW, value_channels)
        else:
            out = self.W[0](out)  # (B, value_channels, HW)
            out = self.W[1](out.permute(0, 2, 1))  # (B, HW, value_channels) and is output of layer norm

        return out


class LangProject(nn.Module):
    def __init__(self, l_in_channels, l_out_channels):
        super(LangProject, self).__init__()
        self.l_in_channels = l_in_channels
        self.l_out_channels = l_out_channels
        self.project = nn.Sequential(
            nn.Linear(self.l_in_channels, self.l_out_channels),
            nn.ReLU(),
            nn.Linear(self.l_out_channels, self.l_out_channels),
        #    nn.BatchNorm1d(self.l_out_channels)
        )

    def forward(self, x, l, l_mask):
        # x shape: not used; keep interface consistent
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)

        ############################
        # language embeddings average pooling
        ############################
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        l = (l * l_mask).sum(dim=-1).div(l_mask.sum(dim=-1))  # (B, l_in_channels)
        ############################

        ############################
        # language embedding projection
        ############################
        return self.project(l).unsqueeze(1)  # (B, 1, l_out_channels) l_out_channels is value_channels


#######################################################################
# The original image Swin architecture; the layer and the final model #
#######################################################################
# a stage
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
        # output of a Block has shape (B, H*W, dim)


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            #load_checkpoint(self, pretrained, strict=False, logger=logger)
            load_checkpoint(self, pretrained, strict=('upernet' in pretrained), logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)  # output of a Block has shape (B, H*W, dim)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
