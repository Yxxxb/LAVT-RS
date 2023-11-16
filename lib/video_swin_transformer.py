import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
# from .mmcv_custom import load_checkpoint
# from mmseg.utils import get_root_logger
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from typing import Optional




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
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
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
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

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

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class MMBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 num_heads_fusion=1,
                 fusion_drop=0.0,
                 sr_ratio=1,
                 args=None
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.version = args.version
        self.fuse = args.fuse
        self.hs = args.hs
        self.lazy_pred = args.lazy_pred
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

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        # fuse before downsampling
        if self.ts_pwam:
            self.fusion = TSPWAM(dim,  # both the visual input and for combining, num of channels
                                 dim,  # v_in
                                 768,  # l_in
                                 dim,  # key
                                 dim,  # value
                                 num_heads=num_heads_fusion,
                                 dropout=fusion_drop,
                                 attention=(args.fuse != 'simple'),
                                 conv3d_kernel_size=conv3d_kernel_size,
                                 sum=args.tspwam_sum,
                                 w_3x3=args.w_3x3,
                                 mm_3x3=args.mm_3x3,
                                 cat_reduce_3=args.cat_reduce_3
                                 )
        elif self.t_pwam:
            self.fusion = TPWAM(dim,  # both the visual input and for combining, num of channels
                                dim,  # v_in
                                768,  # l_in
                                dim,  # key
                                dim,  # value
                                num_heads=num_heads_fusion,
                                dropout=fusion_drop,
                                attention=(args.fuse != 'simple'),
                                conv3d_kernel_size=conv3d_kernel_size
                                )
        elif self.t_pwam_comp:
            self.fusion = TPWAMComp(dim,  # both the visual input and for combining, num of channels
                                    dim,  # v_in
                                    768,  # l_in
                                    dim,  # key
                                    dim,  # value
                                    num_heads=num_heads_fusion,
                                    dropout=fusion_drop,
                                    attention=(args.fuse != 'simple'),
                                    conv3d_kernel_size=conv3d_kernel_size
                                    )
        elif self.sep_t_pwam:
            self.fusion = SepTPWAM(dim, dim, 768, dim, dim, num_heads=num_heads_fusion, dropout=fusion_drop,
                                   conv3d_kernel_size_t=conv3d_kernel_size_t, conv3d_kernel_size_s=conv3d_kernel_size_s,
                                   w_3x3=args.w_3x3, mm_3x3=args.mm_3x3, w_3=args.w_3, mm_3=args.mm_3,
                                   sum_3_kernel_size=sept_sum_3_kernel_size,
                                   cat_reduce_kernel_size=sept_cat_reduce_kernel_size,
                                   w_t3x3_s1x1=args.w_t3x3_s1x1, mm_t3x3_s1x1=args.mm_t3x3_s1x1,
                                   args=args
                                   )

        elif self.sep_t_pwam_inner:
            self.fusion = SepTPWAMInner(dim, dim, 768, dim, dim, num_heads=num_heads_fusion, dropout=fusion_drop,
                                        conv3d_kernel_size_t=conv3d_kernel_size_t, conv3d_kernel_size_s=conv3d_kernel_size_s
                                        )

        elif self.seq_t_pwam:
            self.fusion = SeqTPWAM(dim, dim, 768, dim, dim, num_heads=num_heads_fusion, dropout=fusion_drop,
                                   conv3d_kernel_size_t=conv3d_kernel_size_t, conv3d_kernel_size_s=conv3d_kernel_size_s,
                                   res=args.res
                                   )
        elif self.sep_seq_t_pwam:
            self.fusion = SepSeqTPWAM(dim, dim, 768, dim, dim, num_heads=num_heads_fusion, dropout=fusion_drop,
                                      conv3d_kernel_size_t=conv3d_kernel_size_t,
                                      conv3d_kernel_size_s=conv3d_kernel_size_s,
                                      conv3d_kernel_size_sq=conv3d_kernel_size_sq, res=args.res
                                      )
        elif self.sep_seq_t_pwam_inner:
            self.fusion = SepSeqTPWAMInner(dim, dim, 768, dim, dim, num_heads=num_heads_fusion, dropout=fusion_drop,
                                           conv3d_kernel_size_t=conv3d_kernel_size_t,
                                           conv3d_kernel_size_s=conv3d_kernel_size_s,
                                           conv3d_kernel_size_sq=conv3d_kernel_size_sq, res=args.res
                                           )
        else:
            self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                               dim,  # v_in
                               768,  # l_in
                               dim,  # key
                               dim,  # value
                               num_heads=num_heads_fusion,
                               dropout=fusion_drop,
                               attention=(args.fuse != 'simple')
                               )

        if self.version != 'default':
            pass
        elif self.is_last_layer and self.use_checkpoint:
            print('Last stage and using checkpoint; not declaring LG!')
            pass
        else:
            self.res_gate = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.ReLU(),
                nn.Linear(dim, dim, bias=False),
                nn.Tanh()
                #    nn.Sigmoid()
            )

        # patch merging layer
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

        if self.version != 'default' or (self.is_last_layer and self.use_checkpoint):
            return
        # initialize the gates to 0
        nn.init.zeros_(self.res_gate[0].weight)
        nn.init.zeros_(self.res_gate[2].weight)

    def forward(self, x, l, l_mask):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)  # b d h w c

        if self.lazy_pred:
            x_out = rearrange(x, 'b d h w c -> b c d h w')  # features before fusion and before downsampling
            # This is V_i in Fig. 4

        # PWAM fusion
        x = x.view(B, D*H*W, C)  # b d*h*w c; downsampling only happens at layer level; block-level is safe
        if self.ts_pwam or self.t_pwam or self.sep_t_pwam or self.seq_t_pwam or self.sep_t_pwam_inner \
                or self.sep_seq_t_pwam or self.sep_seq_t_pwam_inner or self.t_pwam_comp:
            x_residual = self.fusion(x.view(B, D, H, W, C), l, l_mask)  # the output has shape (B, DHW, C)
        else:
            x_residual = self.fusion(x, l, l_mask)  # both input and output have shape (B, DHW, C)
        # from inside the PWAM module, the shape is indeed (B, T*H*W, dim)
        if self.version == "default" and (not self.use_checkpoint or not self.is_last_layer):
            # apply a gate on the residual
            x = x + (self.res_gate(x_residual) * x_residual)  # have shape (B, DHW, C)
        elif self.version == "no_gate":
            x = x + x_residual  # have shape (B, DHW, C)

        x_residual = x_residual.view(B, D, H, W, C)  # b d h w c
        x = x.view(B, D, H, W, C)  # b d h w c

        if self.hs:
            x_out = rearrange(x, 'b d h w c -> b c d h w')  # features after fusion and before downsampling
            # This is E_i in Fig. 4

        if self.downsample is not None:
            x = self.downsample(x)  # input (B, D, H, W, C); output (B, D, H/2, W/2, 2C)
        x_residual = rearrange(x_residual, 'b d h w c -> b c d h w')
        x = rearrange(x, 'b d h w c -> b c d h w')

        if self.hs or self.lazy_pred:
            return x_out, x

        return x_residual, x
        # depending on if x has been downsampled, can have shape b c d h w or b 2c h/2 w/2;
        # x_residual always has the shape b c d h w


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class MultiModalSwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=False,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.0,
                 args=None
                 ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MMBasicLayer(
                dim=int(embed_dim * 2**i_layer),
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
                downsample=PatchMerging if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint,
                num_heads_fusion=num_heads_fusion[i_layer],
                fusion_drop=fusion_drop,
                sr_ratio=sr_ratio[min(i_layer, len(sr_ratio) - 1)],
                args=args
            )
            self.layers.append(layer)

        # adding an output norm layer is the pratice of segmentation swin. we follow that here.
        # self.num_features = int(embed_dim * 2**(self.num_layers-1))
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        # self.norm = norm_layer(self.num_features)
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

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            no args
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

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

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            # For our VOSRE task we just assume the pre-trained weights are from a 3D model.
            # Thus excluding the option of starting training from 2D weights...
            # Directly load 3D model.
            # load_checkpoint(self, self.pretrained, strict=False, logger=logger)
            # a migration from MTTR and ReferFormer
            state_dict = torch.load(self.pretrained)['state_dict']
            # extract video swin's 3D pre-trained weights and ignore the rest (prediction head etc.)
            state_dict = {k[9:]: v for k, v in state_dict.items() if 'backbone.' in k}
            # sum over the patch embedding weight temporal dim, e.g., for tiny: [96, 3, 2, 4, 4] --> [96, 3, 1, 4, 4]
            patch_embed_weight = state_dict['patch_embed.proj.weight']
            patch_embed_weight = patch_embed_weight.sum(dim=2, keepdims=True)
            state_dict['patch_embed.proj.weight'] = patch_embed_weight
            self.load_state_dict(state_dict, strict=False)  # strict=False is because
            # PWAM (fusion) and res_gate parameters are new and not included in pre-trained weights, and
            # we do not make use of the "norm" parameters in pre-trained weights
            # their "norm" (weight and bias) is applied on the fourth-stage output (which is purely visual);
            # we use per-stage norm, i.e., "norm1", ..., "norm4" and our features are multi-modal.
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, l, l_mask):
        """Forward function."""
        # x shape: (B, 3, T, H, W)
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        outs = []
        #for layer in self.layers:
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x.contiguous(), l, l_mask)
            # both have shape b c d h w; these are layer outputs
            # both x_out and x should have shape: n c d h w

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = rearrange(x_out, 'n c d h w -> n d h w c')  # layernorm last dim should be c; others are batch dims
                x_out = norm_layer(x_out)  # n d h w c
                x_out = rearrange(x_out, 'n d h w c -> (n d) c h w')  # (B*T, Ci, Hi, Wi)
                outs.append(x_out)  # (B*T, Ci, Hi, Wi)

        #x = rearrange(x, 'n c d h w -> n d h w c')
        #x = self.norm(x)
        #x = rearrange(x, 'n d h w c -> n c d h w')

        #return x
        return tuple(outs)  # 4-tuple, outs[0] lowest level; outs[3] highest level, each: (B*T, Ci, Hi, Wi)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MultiModalSwinTransformer3D, self).train(mode)
        self._freeze_stages()


class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 attention=True):
        super(PWAM, self).__init__()
        # input x shape: (B, T*H*W, dim); has an additional T or (D) dimension compared to the previous one
        # simply multiply H*W with T, just roll T into H*W to form a single dimension. Thought about this,
        # this is still valid for the PWAM formulation
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        self.attention = attention
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                         )
        if attention:
            self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                                l_in_channels,  # l_in
                                                                key_channels,  # key
                                                                value_channels,  # value
                                                                out_channels=value_channels,  # out
                                                                num_heads=num_heads)
        else:
            self.image_lang_att = LangProject(l_in_channels=l_in_channels, l_out_channels=value_channels)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, T*H*W, dim)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, T*H*W)

        lang = self.image_lang_att(x, l, l_mask)  # (B, T*H*W, dim) or (B, 1, dim) if ablation

        lang = lang.permute(0, 2, 1)  # (B, dim, T*H*W) or (B, dim, 1) if ablation

        mm = torch.mul(vis, lang)  # (B, dim, T*H*W)
        mm = self.project_mm(mm)  # (B, dim, T*H*W)

        mm = mm.permute(0, 2, 1)  # (B, T*H*W, dim)

        return mm


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, T*H*W, v_in_channels)
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

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, T*H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, T*H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, THW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, v_in_channels, T*H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, T*H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, T*H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, THW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, T*H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, T*H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, t*h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, T*H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, THW, self.value_channels)  # (B, T*H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, THW)
        out = self.W(out)  # (B, value_channels, THW)
        out = out.permute(0, 2, 1)  # (B, THW, value_channels)

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
        l = (l * l_mask).sum(dim=-1).div(l_mask.sum(dim=-1))  # (B, l_out_channels)
        ############################

        ############################
        # language embedding projection
        ############################
        return self.project(l).unsqueeze(1)  # (B, 1, l_out_channels) l_out_channels is value_channels


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


class TSPWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 attention=True, conv3d_kernel_size=(3, 1, 1), sum=False, w_3x3=False, mm_3x3=False,
                 cat_reduce_3=False):
        super(TSPWAM, self).__init__()
        # input x shape: (B, D, H, W, C)
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        self.attention = attention
        self.sum = sum  # Boolean
        self.w_3x3 = w_3x3
        self.mm_3x3 = mm_3x3
        self.cat_reduce_3 = cat_reduce_3
        self.conv3d_kernel_size = conv3d_kernel_size  # (depth, height, width)
        self.padding = _3d_kernel_size_to_padding_size(self.conv3d_kernel_size)  # 1, 1, 1 or 1, 0, 0

        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                         )
        self.temporal_vis_project = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=self.conv3d_kernel_size,
                                                            stride=1, padding=self.padding),
                                                  nn.GELU(),
                                                  nn.Dropout(dropout)
                                                  )

        if attention:
            self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                                l_in_channels,  # l_in
                                                                key_channels,  # key
                                                                value_channels,  # value
                                                                out_channels=value_channels,  # out
                                                                num_heads=num_heads)
            self.temporal_image_lang_att = TemporalSpatialImageLanguageAttention(v_in_channels,  # v_in
                                                                l_in_channels,  # l_in
                                                                key_channels,  # key
                                                                value_channels,  # value
                                                                out_channels=value_channels,  # out
                                                                num_heads=num_heads,
                                                                conv3d_kernel_size=self.conv3d_kernel_size,
                                                                complete=self.w_3x3
                                                                )
        else:
            self.image_lang_att = LangProject(l_in_channels=l_in_channels, l_out_channels=value_channels)
            self.temporal_image_lang_att = LangProject(l_in_channels=l_in_channels, l_out_channels=value_channels)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

        if self.mm_3x3:
            self.project_temporal_mm = nn.Sequential(nn.Conv3d(value_channels, value_channels,
                                                               kernel_size=self.conv3d_kernel_size,
                                                               stride=1, padding=self.padding),
                                                     nn.GELU(),
                                                     nn.Dropout(dropout)
                                                     )
        else:
            self.project_temporal_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                                     nn.GELU(),
                                                     nn.Dropout(dropout)
                                                     )
        if not self.sum:
            # New function added for fusing the two types of features
            if self.cat_reduce_3:
                self.out_reduce = nn.Sequential(nn.Conv3d(2*value_channels, value_channels,
                                                          kernel_size=(1, 3, 3),
                                                          stride=1, padding=(0, 1, 1)
                                                          ),
                                                          nn.GELU(),
                                                          nn.Dropout(dropout)
                                                )
            else:
                self.out_reduce = nn.Sequential(
                    nn.Linear(2*value_channels, value_channels),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )

    def forward(self, x, l, l_mask):
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, D, H, W, C = x.size()
        vis = self.vis_project(x.view(B, D*H*W, C).permute(0, 2, 1))  # (B, dim, T*H*W)
        temporal_vis = self.temporal_vis_project(x.permute(0, 4, 1, 2, 3))  # (B, dim, T, H, W)
        temporal_vis = temporal_vis.view(B, C, D*H*W)  # to make shapes consistent

        lang = self.image_lang_att(x.view(B, D*H*W, C), l, l_mask)  # (B, T*H*W, dim) or (B, 1, dim) if ablation
        temporal_lang = self.temporal_image_lang_att(x, l, l_mask)  # (B, T*H*W, dim) or (B, 1, dim) if ablation

        lang = lang.permute(0, 2, 1)  # (B, dim, T*H*W) or (B, dim, 1) if ablation
        temporal_lang = temporal_lang.permute(0, 2, 1)  # (B, dim, T*H*W) or (B, dim, 1) if ablation

        mm = torch.mul(vis, lang)  # (B, dim, T*H*W)
        temporal_mm = torch.mul(temporal_vis, temporal_lang)  # (B, dim, T*H*W)

        mm = self.project_mm(mm)  # (B, dim, T*H*W)
        mm = mm.permute(0, 2, 1)  # (B, T*H*W, dim)

        if self.mm_3x3:
            temporal_mm = temporal_mm.view(B, C, D, H, W)  # (B, dim, T, H, W)
            temporal_mm = self.project_temporal_mm(temporal_mm)  # (B, dim, T, H, W)
            temporal_mm = temporal_mm.view(B, C, D*H*W)  # (B, dim, T*H*W)
            temporal_mm = temporal_mm.permute(0, 2, 1)  # (B, T*H*W, dim)
        else:
            temporal_mm = self.project_temporal_mm(temporal_mm)  # (B, dim, T*H*W)
            temporal_mm = temporal_mm.permute(0, 2, 1)  # (B, T*H*W, dim)

        if self.sum:
            out = mm + temporal_mm
        else:  # concatenation
            out = torch.cat([mm, temporal_mm], dim=-1)  # (B, T*H*W, 2*dim)
            if self.cat_reduce_3:
                out = out.permute(0, 2, 1).view(B, 2*C, D, H, W)  # (B, 2*C, D, H, W)
                out = self.out_reduce(out)  # (B, C, D, H, W)
                out = out.view(B, C, D*H*W).permute(0, 2, 1)  # (B, T*H*W, dim)
            else:
                out = self.out_reduce(out)  # (B, T*H*W, dim)

        return out


class TPWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 attention=True, conv3d_kernel_size=(3, 1, 1)):
        super(TPWAM, self).__init__()
        # input x shape: (B, D, H, W, C)
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        self.attention = attention
        self.conv3d_kernel_size = conv3d_kernel_size  # (depth, height, width)
        self.padding = _3d_kernel_size_to_padding_size(self.conv3d_kernel_size)  # 1, 1, 1 or 1, 0, 0

        self.temporal_vis_project = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=self.conv3d_kernel_size,
                                                            stride=1, padding=self.padding),
                                                  nn.GELU(),
                                                  nn.Dropout(dropout)
                                                  )

        if attention:
            self.temporal_image_lang_att = TemporalSpatialImageLanguageAttention(v_in_channels,  # v_in
                                                                l_in_channels,  # l_in
                                                                key_channels,  # key
                                                                value_channels,  # value
                                                                out_channels=value_channels,  # out
                                                                num_heads=num_heads,
                                                                conv3d_kernel_size=self.conv3d_kernel_size
                                                                )
        else:
            self.temporal_image_lang_att = LangProject(l_in_channels=l_in_channels, l_out_channels=value_channels)

        self.project_temporal_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                                 nn.GELU(),
                                                 nn.Dropout(dropout)
                                                 )

    def forward(self, x, l, l_mask):
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, D, H, W, C = x.size()
        temporal_vis = self.temporal_vis_project(x.permute(0, 4, 1, 2, 3))  # (B, dim, T, H, W)
        temporal_vis = temporal_vis.view(B, C, D*H*W)  # to make shapes consistent

        temporal_lang = self.temporal_image_lang_att(x, l, l_mask)  # (B, T*H*W, dim) or (B, 1, dim) if ablation
        temporal_lang = temporal_lang.permute(0, 2, 1)  # (B, dim, T*H*W) or (B, dim, 1) if ablation

        temporal_mm = torch.mul(temporal_vis, temporal_lang)  # (B, dim, T*H*W)
        temporal_mm = self.project_temporal_mm(temporal_mm)  # (B, dim, T*H*W)
        temporal_mm = temporal_mm.permute(0, 2, 1)  # (B, T*H*W, dim)

        return temporal_mm


class TPWAMComp(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 attention=True, conv3d_kernel_size=(3, 1, 1)):
        super(TPWAMComp, self).__init__()
        # input x shape: (B, D, H, W, C)
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        self.attention = attention
        self.conv3d_kernel_size = conv3d_kernel_size  # (depth, height, width)
        self.padding = _3d_kernel_size_to_padding_size(self.conv3d_kernel_size)  # 1, 1, 1 or 1, 0, 0

        self.temporal_vis_project = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=self.conv3d_kernel_size,
                                                            stride=1, padding=self.padding),
                                                  nn.GELU(),
                                                  nn.Dropout(dropout)
                                                  )

        if attention:
            self.temporal_image_lang_att = TemporalSpatialImageLanguageAttention(v_in_channels,  # v_in
                                                                                 l_in_channels,  # l_in
                                                                                 key_channels,  # key
                                                                                 value_channels,  # value
                                                                                 out_channels=value_channels,  # out
                                                                                 num_heads=num_heads,
                                                                                 conv3d_kernel_size=self.conv3d_kernel_size,
                                                                                 complete=True
                                                                                 )
        else:
            self.temporal_image_lang_att = LangProject(l_in_channels=l_in_channels, l_out_channels=value_channels)

        # Change: last mm projection is also 3D conv in this version
        self.project_temporal_mm = nn.Sequential(nn.Conv3d(value_channels, value_channels,
                                                           kernel_size=self.conv3d_kernel_size,
                                                           stride=1, padding=self.padding),
                                                 nn.GELU(),
                                                 nn.Dropout(dropout)
                                                 )

    def forward(self, x, l, l_mask):
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, D, H, W, C = x.size()
        temporal_vis = self.temporal_vis_project(x.permute(0, 4, 1, 2, 3))  # (B, dim, T, H, W)
        temporal_vis = temporal_vis.view(B, C, D*H*W)  # to make shapes consistent

        temporal_lang = self.temporal_image_lang_att(x, l, l_mask)  # (B, T*H*W, dim) or (B, 1, dim) if ablation
        temporal_lang = temporal_lang.permute(0, 2, 1)  # (B, dim, T*H*W) or (B, dim, 1) if ablation

        temporal_mm = torch.mul(temporal_vis, temporal_lang)  # (B, dim, T*H*W)

        # input to 3D conv needs shape (B, dim, T, H, W)
        temporal_mm = temporal_mm.reshape(B, C, D, H, W)  # (B, dim, T, H, W)
        temporal_mm = self.project_temporal_mm(temporal_mm)  # (B, dim, T, H, W)
        temporal_mm = temporal_mm.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, dim)
        temporal_mm = temporal_mm.reshape(B, D*H*W, C)  # (B, T*H*W, dim)

        return temporal_mm


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


class SepTPWAMInner(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 conv3d_kernel_size_t=(3, 1, 1), conv3d_kernel_size_s=(1, 1, 1)):
        super(SepTPWAMInner, self).__init__()
        # input x shape: (B, D, H, W, C)
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        # conv3d_kernel_size  # (depth, height, width)

        self.num_heads = num_heads
        padding_t = _3d_kernel_size_to_padding_size(conv3d_kernel_size_t)  # 3 -> 1; 1 -> 0
        padding_s = _3d_kernel_size_to_padding_size(conv3d_kernel_size_s)  # 3 -> 1; 1 -> 0
        out_channels = value_channels

        # The outer layer projection; note this one does not have a temporal projection
        self.spatial_vis_project = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0),
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

        # wrap-up projection
        self.W = nn.Sequential(
            nn.Conv1d(value_channels, out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(out_channels),
        )

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
        spatial_vis = self.spatial_vis_project(x)  # (B, dim, T, H, W)
        spatial_vis = spatial_vis.view(B, C, D * H * W)  # (B, dim, THW) to make shapes consistent

        #### temporal spatial image language attention forward transferred here
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        query_t = self.f_query_t(x)  # (B, key_channels, T, H, W) because Conv3D
        query_s = self.f_query_s(x)  # (B, key_channels, T, H, W) because Conv3D
        query = query_t + query_s  # (B, key_channels, T, H, W)

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
        ts_lang = self.W(ts_lang)  # (B, C, THW)
        # the above

        mm = torch.mul(spatial_vis, ts_lang)  # (B, dim, T*H*W)
        mm = self.project_mm(mm)  # (B, dim, T*H*W)
        mm = mm.permute(0, 2, 1)  # (B, T*H*W, dim)

        return mm


class SeqTPWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 conv3d_kernel_size_t=(3, 1, 1), conv3d_kernel_size_s=(1, 1, 1), res=False):
        super(SeqTPWAM, self).__init__()
        # input x shape: (B, D, H, W, C)
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        # conv3d_kernel_size  # (depth, height, width)

        # spatial first, sent to temporal second

        self.res = res  # if true, use P3D-C, i.e., add a residual connection from s to the output of s->t
        self.num_heads = num_heads
        padding_t = _3d_kernel_size_to_padding_size(conv3d_kernel_size_t)  # 3 -> 1; 1 -> 0
        padding_s = _3d_kernel_size_to_padding_size(conv3d_kernel_size_s)  # 3 -> 1; 1 -> 0
        out_channels = value_channels

        # The outer layer; top red square; visual projection outside of attention and elementwise multiplication
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

        # Queries: visual features; t for temporal s for spatial; sequentially compose the two s -> t
        self.f_query_t = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_t, stride=1, padding=padding_t),
            nn.InstanceNorm3d(key_channels),
        )

        self.f_query_s = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_s, stride=1, padding=padding_s),
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

        # wrap-up projection
        self.W = nn.Sequential(
            nn.Conv1d(value_channels, out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(out_channels),
        )

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
        spatial_vis = self.spatial_vis_project(x)  # (B, dim, T, H, W)
        ts_vis = self.temporal_vis_project(spatial_vis)  # (B, dim, T, H, W)
        ts_vis = ts_vis.view(B, C, D * H * W)  # (B, dim, THW) to make shapes consistent

        if self.res:
            ts_vis = spatial_vis.view(B, C, D * H * W) + ts_vis  # sum s as residual; (B, dim, THW)

        #### temporal spatial image language attention forward transferred here
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        query_s = self.f_query_s(x)  # (B, key_channels, T, H, W) because Conv3D
        query = self.f_query_t(query_s)  # (B, key_channels, T, H, W) because Conv3D
        if self.res:
            query = query_s + query  # (B, key_channels, T, H, W)

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
        ts_lang = self.W(ts_lang)  # (B, C, THW)
        # the above

        mm = torch.mul(ts_vis, ts_lang)  # (B, dim, T*H*W)
        mm = self.project_mm(mm)  # (B, dim, T*H*W)
        mm = mm.permute(0, 2, 1)  # (B, T*H*W, dim)

        return mm


class SepSeqTPWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 conv3d_kernel_size_t=(3, 1, 1), conv3d_kernel_size_s=(1, 1, 1), conv3d_kernel_size_sq=(1, 1, 1),
                 res=False):
        super(SepSeqTPWAM, self).__init__()
        # input x shape: (B, D, H, W, C)
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        # conv3d_kernel_size  # (depth, height, width)

        self.num_heads = num_heads
        self.res = res  # if true, use P3D-C, i.e., add a residual connection from sq to the output of sq->tq
        padding_t = _3d_kernel_size_to_padding_size(conv3d_kernel_size_t)  # 3 -> 1; 1 -> 0
        padding_s_seq = _3d_kernel_size_to_padding_size(conv3d_kernel_size_sq)  # 3 -> 1; 1 -> 0
        padding_s = _3d_kernel_size_to_padding_size(conv3d_kernel_size_s)  # 3 -> 1; 1 -> 0

        # inner seq branch has one additional spatial convolution because we decouple t and s in seq branch

        out_channels = value_channels

        # The outer layer; top red square
        self.temporal_vis_project_q = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=conv3d_kernel_size_t,
                                                              stride=1, padding=padding_t),
                                                    nn.GELU(),
                                                    nn.Dropout(dropout)
                                                    )

        self.spatial_vis_project_q = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=conv3d_kernel_size_sq,
                                                              stride=1, padding=padding_s_seq),
                                                   nn.GELU(),
                                                   nn.Dropout(dropout)
                                                   )
        # the above is decoupled 3d conv with one temporal conv one spatial conv

        self.spatial_vis_project = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=conv3d_kernel_size_s,
                                                              stride=1, padding=padding_s),
                                                 nn.GELU(),
                                                 nn.Dropout(dropout)
                                                 )

        # Queries: visual features t for temporal s for spatial, sum the two
        self.f_query_t_q = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_t, stride=1, padding=padding_t),
            nn.InstanceNorm3d(key_channels),
        )
        self.f_query_s_q = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_sq, stride=1, padding=padding_s_seq),
            nn.InstanceNorm3d(key_channels),
        )
        # the above is decoupled 3d conv with one temporal conv one spatial conv

        self.f_query_s = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_s, stride=1, padding=padding_s),
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

        # wrap-up projection
        self.W = nn.Sequential(
            nn.Conv1d(value_channels, out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(out_channels),
        )

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
        spatial_vis_seq = self.spatial_vis_project_q(x)  # (B, C, D, H, W)
        temporal_vis = self.temporal_vis_project_q(spatial_vis_seq)  # (B, dim, T, H, W)
        if self.res:
            temporal_vis = spatial_vis_seq + temporal_vis  # sum s as residual; (B, dim, T, H, W)

        spatial_vis = self.spatial_vis_project(x)  # (B, dim, T, H, W)

        ts_vis = temporal_vis + spatial_vis  # sum to fuse; (B, dim, THW)
        ts_vis = ts_vis.view(B, C, D * H * W)  # (B, dim, THW) to make shapes consistent

        #### temporal spatial image language attention forward transferred here
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        query_s_q = self.f_query_s_q(x)  # (B, C, D, H, W)
        query_t = self.f_query_t_q(query_s_q)  # (B, key_channels, T, H, W) because Conv3D
        if self.res:
            query_t = query_s_q + query_t  # (B, key_channels, T, H, W)

        query_s = self.f_query_s(x)  # (B, key_channels, T, H, W) because Conv3D
        query = query_t + query_s  # (B, key_channels, T, H, W)

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
        ts_lang = self.W(ts_lang)  # (B, C, THW)
        # the above

        mm = torch.mul(ts_vis, ts_lang)  # (B, dim, T*H*W)
        mm = self.project_mm(mm)  # (B, dim, T*H*W)
        mm = mm.permute(0, 2, 1)  # (B, T*H*W, dim)

        return mm


class SepSeqTPWAMInner(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0,
                 conv3d_kernel_size_t=(3, 1, 1), conv3d_kernel_size_s=(1, 1, 1), conv3d_kernel_size_sq=(1, 1, 1),
                 res=False):
        super(SepSeqTPWAMInner, self).__init__()
        # input x shape: (B, D, H, W, C)
        # We keep Conv1D here (not using Linear) because our default norm in this module is InstanceNorm1D
        # A note: BatchNorm1D is consistent with InstanceNorm1D,
        # but LayerNorm would require us permuting the feature dimension to the last position
        # conv3d_kernel_size  # (depth, height, width)

        self.num_heads = num_heads
        self.res = res  # if true, use P3D-C, i.e., add a residual connection from sq to the output of sq->tq
        padding_t = _3d_kernel_size_to_padding_size(conv3d_kernel_size_t)  # 3 -> 1; 1 -> 0
        padding_s_seq = _3d_kernel_size_to_padding_size(conv3d_kernel_size_sq)  # 3 -> 1; 1 -> 0
        padding_s = _3d_kernel_size_to_padding_size(conv3d_kernel_size_s)  # 3 -> 1; 1 -> 0

        # inner seq branch has one additional spatial convolution because we decouple t and s in seq branch

        out_channels = value_channels

        # The outer layer projection; note for INNER models the outer layer does not have a temporal projection
        self.spatial_vis_project = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0),
                                                 nn.GELU(),
                                                 nn.Dropout(dropout)
                                                 )

        # Queries: visual features t for temporal s for spatial, sum the two
        self.f_query_t_q = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_t, stride=1, padding=padding_t),
            nn.InstanceNorm3d(key_channels),
        )
        self.f_query_s_q = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_sq, stride=1, padding=padding_s_seq),
            nn.InstanceNorm3d(key_channels),
        )
        # the above is decoupled 3d conv with one temporal conv one spatial conv

        self.f_query_s = nn.Sequential(
            nn.Conv3d(v_in_channels, key_channels, kernel_size=conv3d_kernel_size_s, stride=1, padding=padding_s),
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

        # wrap-up projection
        self.W = nn.Sequential(
            nn.Conv1d(value_channels, out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(out_channels),
        )

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

        spatial_vis = self.spatial_vis_project(x)  # (B, dim, T, H, W)
        spatial_vis = spatial_vis.view(B, C, D * H * W)  # (B, dim, THW) to make shapes consistent

        #### temporal spatial image language attention forward transferred here
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        query_s_q = self.f_query_s_q(x)  # (B, C, D, H, W)
        query_t = self.f_query_t_q(query_s_q)  # (B, key_channels, T, H, W) because Conv3D
        if self.res:
            query_t = query_s_q + query_t  # (B, key_channels, T, H, W)

        query_s = self.f_query_s(x)  # (B, key_channels, T, H, W) because Conv3D
        query = query_t + query_s  # (B, key_channels, T, H, W)

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
        ts_lang = self.W(ts_lang)  # (B, C, THW)
        # the above

        mm = torch.mul(spatial_vis, ts_lang)  # (B, dim, T*H*W)
        mm = self.project_mm(mm)  # (B, dim, T*H*W)
        mm = mm.permute(0, 2, 1)  # (B, T*H*W, dim)

        return mm


class TemporalSpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1,
                 conv3d_kernel_size=(3, 1, 1), complete=False):
        super(TemporalSpatialImageLanguageAttention, self).__init__()
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        self.conv3d_kernel_size = conv3d_kernel_size
        self.padding = _3d_kernel_size_to_padding_size(self.conv3d_kernel_size)  # 1, 1, 1 or 1, 0, 0
        self.complete = complete  # if true, use 3D conv also for the output projection

        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, T*H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv3d(self.v_in_channels, self.key_channels, kernel_size=self.conv3d_kernel_size, stride=1,
                      padding=self.padding),
            nn.InstanceNorm3d(self.key_channels),
        )
        # input shape: (N, C, D, H, W) or (C, D, H, W)
        #dim, dim, kernel_size=self.conv3d_kernel_size, stride=1, padding=self.padding
        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        if not self.complete:
            self.W = nn.Sequential(
                nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
                nn.InstanceNorm1d(self.out_channels),
            )
        else:
            self.W = nn.Sequential(
                nn.Conv3d(self.value_channels, self.out_channels, kernel_size=self.conv3d_kernel_size, stride=1,
                          padding=self.padding),
                nn.InstanceNorm3d(self.out_channels),
            )

    def forward(self, x, l, l_mask):
        # input x shape: (B, D, H, W, C)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, D, H, W, C = x.size()
        # x = x.permute(0, 2, 1)  # (B, v_in_channels, T*H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        query = self.f_query(x)  # (B, key_channels, T, H, W) because Conv3D
        query = query.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        query = query.view(B, D*H*W, self.key_channels)  # (B, DHW, key_channels)
        # query = query.permute(0, 2, 1)  # (B, T*H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, D*H*W, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, T*H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, T*H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, t*h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, T*H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, D*H*W, self.value_channels)  # (B, T*H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, THW)
        if self.complete:
            out = out.reshape(B, self.value_channels, D, H, W)  # (B, value_channels, T, H, W)
            out = self.W(out)  # (B, out_channels, T, H, W) if conv3d
            out = out.permute(0, 2, 3, 4, 1).contiguous().reshape(B, D*H*W, self.out_channels)  # (B, THW, out_channels)
        else:
            out = self.W(out)  # (B, out_channels, THW) if conv1d
            out = out.permute(0, 2, 1)  # (B, THW, out_channels)

        return out


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output