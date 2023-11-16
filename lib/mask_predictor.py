import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from .backbone import LangProject
# 解码框架
class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, args, factor=2):
        super(SimpleDecoding, self).__init__()

        self.lazy_pred = args.lazy_pred
        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        if not self.lazy_pred:
            self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
            self.bn1_2 = nn.BatchNorm2d(hidden_size)
            self.relu1_2 = nn.ReLU()
            self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
            self.bn2_2 = nn.BatchNorm2d(hidden_size)
            self.relu2_2 = nn.ReLU()

        if args.interpolate_before_seg:
            self.conv2_1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
            self.bn1_1 = nn.BatchNorm2d(hidden_size)
            self.relu1_1 = nn.ReLU()

        if args.seg_last:
            self.conv1_0 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
            self.bn1_0 = nn.BatchNorm2d(hidden_size)
            self.relu1_0 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

        self.interpolate_before_seg = args.interpolate_before_seg
        self.seg_last = args.seg_last
    
    
    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        if not self.lazy_pred:
            # fuse top-down features and Y1 features
            if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
                x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
            x = torch.cat([x, x_c1], dim=1)
            x = self.conv1_2(x)
            x = self.bn1_2(x)
            x = self.relu1_2(x)
            x = self.conv2_2(x)
            x = self.bn2_2(x)
            x = self.relu2_2(x)
        if self.interpolate_before_seg:
            x = F.interpolate(input=x, size=(2 * x_c1.size(-2), 2 * x_c1.size(-1)), mode='bilinear', align_corners=True)
            x = self.conv2_1(x)
            x = self.bn1_1(x)
            x = self.relu1_1(x)
            if self.seg_last:
                x = F.interpolate(input=x, size=(4 * x_c1.size(-2), 4 * x_c1.size(-1)), mode='bilinear', align_corners=True)
                x = self.conv1_0(x)
                x = self.bn1_0(x)
                x = self.relu1_0(x)

        return self.conv1_1(x)


    def forward_feats(self, x_c4, x_c3, x_c2, x_c1):
        feats = []
        c4_feats = x_c4
        feats.append(c4_feats)
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        c3_feats = x
        feats.append(c3_feats)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        c2_feats = x
        feats.append(c2_feats)
        if not self.lazy_pred:
            # fuse top-down features and Y1 features
            if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
                x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
            x = torch.cat([x, x_c1], dim=1)
            x = self.conv1_2(x)
            x = self.bn1_2(x)
            x = self.relu1_2(x)
            x = self.conv2_2(x)
            x = self.bn2_2(x)
            x = self.relu2_2(x)
            c1_feats = x
            feats.append(c1_feats)

        return self.conv1_1(x), feats


class LTSDecoding(nn.Module):
    def __init__(self, c4_dims, args, factor=2):
        super(LTSDecoding, self).__init__()

        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)

        self.lang_gen = LangProject(768, c4_size)  # (B, 1, c4_size)

        self.lang_proj = nn.Sequential(nn.Linear(c4_size, c4_size),
                                       nn.LeakyReLU()
                                       )

        self.vis_proj = nn.Sequential(nn.Conv2d(c4_size, c4_size, 1, 1, bias=False),
                                      nn.LeakyReLU()
                                      )

        self.conv4 = nn.Sequential(nn.Conv2d(c4_size, hidden_size, 1, 1, bias=False),
                                   nn.LeakyReLU()
                                   )

        self.conv3_v = nn.Sequential(nn.Conv2d(c3_size, hidden_size, 1, 1, bias=False),
                                     nn.LeakyReLU()
                                     )

        self.conv3 = nn.Sequential(nn.Conv2d(2*hidden_size, hidden_size, 1, 1, bias=False),
                                   nn.LeakyReLU()
                                   )

        self.conv2_v = nn.Sequential(nn.Conv2d(c2_size, c2_size, 1, 1, bias=False),
                                     nn.LeakyReLU()
                                     )

        self.lang_filter = nn.Linear(c4_size, hidden_size+c2_size)

        self.aspp = ASPP(hidden_size+c2_size+1, hidden_size//2, [12, 24, 36], args)

        self.conv1_1 = nn.Conv2d(hidden_size//2, 2, 1)

        self.last_upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x_c4, x_c3, x_c2, l, l_mask):
        # x_i shape: (b, c_i, h_i, w_i)
        # l shape: (B, 768, N_l)
        # l_mask shape: (B, N_l, 1)

        v = self.vis_proj(x_c4)  # (B, c4_size, h_4, w_4)
        l = self.lang_gen(None, l, l_mask)  # (B, 1, c4_size)
        l = self.lang_proj(l)  # (B, 1, c4_size)
        l = l.squeeze(1)  # (B, c4_size)

        mm = v * l.unsqueeze(-1).unsqueeze(-1)  # (B, c4_size, h_4, w_4)

        mm = F.interpolate(input=mm, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        # (B, c4_size, h_3, w_3)
        mm = self.conv4(mm)  # (B, hidden_size, h_3, w_3)
        x_c3 = self.conv3_v(x_c3)  # (B, hidden_size, h_3, w_3)
        mm = torch.cat([mm, x_c3], dim=1)  # (B, 2*hidden_size, h_3, w_3)

        mm = F.interpolate(input=mm, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        # (B, 2*hidden_size, h_2, w_2)
        mm = self.conv3(mm)  # (B, hidden_size, h_2, w_2)
        x_c2 = self.conv2_v(x_c2)  # (B, c2_size, h_2, w_2)
        mm = torch.cat([mm, x_c2], dim=1)  # (B, hidden_size+c2_size, h_2, w_2)

        l_kernel = self.lang_filter(l)  # (B, hidden_size+c2_size)
        l_kernel = l_kernel.unsqueeze(-1).unsqueeze(-1)  # (B, hidden_size+c2_size, 1, 1)

        relevance_mask = (mm * l_kernel).sum(dim=1, keepdim=True)  # (B, 1, h_2, w_2)

        mm = torch.cat([mm, relevance_mask], dim=1)  # (B, hidden_size+c2_size+1, h_2, w_2)
        mm = self.aspp(mm)  # (B, hidden_size//2, h_2, w_2)
        mm = self.conv1_1(mm)  # (B, 2, h_2, w_2)
        mm = self.last_upsample(mm)  # (B, 2, h_1, w_1)

        return mm


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, args):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(args.fusion_drop))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

