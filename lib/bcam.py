import torch
import torch.nn as nn
import torch.nn.functional as F


### From BRINet ###
class BCAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels):
        super(BCAM, self).__init__()
        print('Initializing BCAM module!!!')
        if dim == 128:
            hw = 120*120  # 96*96
        elif dim == 256:
            hw = 60*60  # 48*48
        elif dim == 512:
            hw = 30*30  # 24*24
        elif dim == 1024:
            hw = 15*15  # 12*12

        self.lang_reduce = nn.Linear(l_in_channels, dim)

        # input x shape: (B, H*W, dim)
        self.vis_1 = nn.Sequential(nn.Linear(v_in_channels, dim),
                                   nn.ReLU()
                                  )
        self.vis_2 = nn.Sequential(nn.Linear(v_in_channels, dim),
                                   nn.ReLU()
                                  )
        self.vis_3 = nn.Sequential(nn.Linear(v_in_channels, dim),
                                   nn.ReLU()
                                  )
        self.vis_4 = nn.Sequential(nn.Linear(v_in_channels, dim),
                                   nn.ReLU()
                                  )

        self.out_1 = nn.Linear(dim, dim)
        self.vis_2_2 = nn.Linear(dim, dim)
        self.a_proj = nn.Linear(dim, hw)
        self.out3_proj = nn.Sequential(nn.Linear(2*dim, dim),
                                       nn.ReLU()
                                       )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        # l = l.permute(0, 2, 1)  # (B, N_l, l_in_channels)
        l = self.lang_reduce(l.permute(0, 2, 1))  # (B, N_l, dim)
        l = l.permute(0, 2, 1)  # (B, dim, N_l)

        # VLAM
        query = self.vis_1(x)  # (B, H*W, dim)
        sim = torch.matmul(query, l)  # (B, HW, N_l)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        sim = sim + (1e4 * l_mask - 1e4)  # assign a very small number to padding positions
        sim = F.softmax(sim, dim=-1)  # (B, HW, N_l)
        out = torch.matmul(sim, l.permute(0, 2, 1))  # (B, HW, dim)

        # LVAM
        query2 = self.vis_2(x)  # (B, H*W, dim)
        A = torch.tanh(self.out_1(out) + self.vis_2_2(query2))  # (B, H*W, dim)
        A = self.a_proj(A)  # (B, H*W, H*W)
        rel_map = F.softmax(A, dim=-1)  # (B, H*W, H*W)
        query3 = self.vis_3(x)  # (B, H*W, dim)
        out2 = torch.matmul(rel_map, query3)  # (B, HW, dim)
        out3 = torch.cat([out2, out], dim=-1)  # # (B, HW, 2*dim)
        out3 = self.out3_proj(out3)  # (B, HW, dim)

        query4 = self.vis_4(x)  # (B, H*W, dim)
        out3 = out3 + query4

        return out3  # (B, H*W, dim)


class GACD(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, num_heads=0):
        super(GACD, self).__init__()
        self.k = num_heads
        self.dim = dim

        print('Initializing GA-CD module!!!')
        self.lang_gen = LangProject(l_in_channels, v_in_channels)  # (B, 1, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.mm_gen = nn.Sequential(nn.Linear(v_in_channels, dim),
                                    nn.ReLU()
                                    )

        # input x shape: (B, H*W, dim)
        self.query = nn.Linear(dim, dim)

        self.key_c = nn.Linear(v_in_channels, dim)

        self.key_d = nn.Linear(v_in_channels, dim)

        self.value = nn.Linear(v_in_channels, dim)

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l = l.permute(0, 2, 1)  # (B, N_l, l_in_channels)

        # Generate sentence-level feature vectors #
        l = self.lang_gen(None, l, l_mask)  # (B, 1, v_in_channels)

        ## Generate MM features ##
        x = l * x  # (B, HW, v_in_channels)
        x = self.mm_gen(x)  # (B, HW, dim)

        ## Grouped attention based on collection-diffusion steps ##
        query = self.query(l)  # (B, 1, dim)
        key_c = self.key_c(x)  # (B, HW, dim)
        key_d = self.key_d(x)  # (B, HW, dim)
        value = self.value(x)  # (B, HW, dim)

        A_c = torch.matmul(query, key_c.permute(0, 2, 1))  # (B, 1, HW)
        A_c = A_c * (self.dim ** -0.5)  # (B, 1, HW)
        A_c = F.softmax(A_c, dim=-1)  # (B, 1, HW)

        A_d = torch.matmul(query, key_d.permute(0, 2, 1))  # (B, 1, HW)
        A_d = A_d * (self.dim ** -0.5)  # (B, 1, HW)
        A_d = torch.sigmoid(A_d)  # (B, 1, HW)

        f_col = torch.matmul(A_c, value)  # (B, 1, dim)
        F_dif = torch.matmul(A_d.permute(0, 2, 1), f_col)  # (B, HW, dim)

        return x + F_dif


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


class EFN(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels):
        super(EFN, self).__init__()
        # input x shape: (B, H*W, dim)
        self.project = nn.Sequential(nn.Conv1d(v_in_channels+l_in_channels, dim, 1, 1),
                                     nn.GELU()
                                    )
        self.lang_project = nn.Sequential(nn.Conv1d(l_in_channels, dim, 1, 1),
                                          nn.GELU()
                                          )

        self.image_lang_att = EFNAttention(in_channels=dim,  # v_in
                                           key_channels=dim,  # key
                                           )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)
        # l: (B, 768, N_l)
        B, HW = x.size(0), x.size(1)

        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        l_sentence = (l * l_mask).sum(dim=-1).div(l_mask.sum(dim=-1))  # (B, 768)
        l_sentence = l_sentence.unsqueeze(-1).expand(B, -1, HW)  # (B, 768, 1)
        x = x.permute(0, 2, 1)  # (B, dim, H*W)
        x = torch.cat([x, l_sentence], dim=1)  # (B, v_in_channels+l_in_channels, H*W)

        M = self.project(x)  # (B, dim, H*W)
        lang = self.lang_project(l)  # (B, dim, N_l)
        lang = lang * l_mask  # (B, dim, N_l)

        score = torch.matmul(M.permute(0, 2, 1), lang)  # (B, H*W, N_l)

        c = M.size(1)  # dim
        score = (c ** -.5) * score  # scaled dot product

        score = score + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        score = F.softmax(score, dim=-1)  # (B, h*w, N_l)
        L = torch.matmul(score, lang.permute(0, 2, 1)).permute(0, 2, 1)  # (B, dim, H*W)

        out = self.image_lang_att(M, L)  # (B, H*W, dim)

        return out


# EFN attention naive version
class EFNAttention(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(EFNAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # Keys: language features, 3D input (B, C_l, #words)
        self.f_key = nn.Sequential(
            nn.Conv1d(self.in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )
        ########################################
        # visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )
        ####################################
        self.W = nn.Sequential(
            nn.Conv1d(2 * self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(self.in_channels),
        )

    def forward(self, M, L):
        # M shape: (B, dim, H*W)
        # L shape: (B, dim, H*W)
        B, c, HW = M.size(0), M.size(1), M.size(2)
        h = w = int(HW**0.5)

        M = self.f_query(M)  # (B, key_channels, H*W)
        L = self.f_key(L)  # (B, key_channels, H*W)
        if HW > 225:
            M = M.view(B, c, h, w)
            L = L.view(B, c, h, w)
            M = self.pool1(M)  # downsample by 2, otherwise footprint too big
            L = self.pool2(L)  # downsample by 2, otherwise footprint too big
            M = M.view(B, c, HW // 4)
            L = L.view(B, c, HW // 4)


        sim_map = torch.matmul(M.permute(0, 2, 1), L)  # (B, H*W, H*W)

        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map_1 = F.softmax(sim_map, dim=-1)  # (B, H*W, H*W)
        sim_map_2 = F.softmax(sim_map, dim=-2).permute(0, 2, 1)  # (B, H*W, H*W)

        Lp = torch.matmul(sim_map_1, L.permute(0, 2, 1))  # (B, H*W, in_channels)
        Mp = torch.matmul(sim_map_2, M.permute(0, 2, 1))  # (B, H*W, in_channels)

        LpMp_cat = torch.cat([Lp, Mp], dim=-1)  # (B, H*W, 2*in_channels)
        LpMp_cat = LpMp_cat.permute(0, 2, 1)  # (B, 2*in_channels, H*W)

        out = self.W(LpMp_cat)  # (B, in_channels, H*W)
        if HW > 225:
            out = out.view(B, c, h//2, w//2)
            out = self.upsample(out)
            out = out.view(B, c, HW)

        out = out.permute(0, 2, 1)  # (B, H*W, in_channels)

        return out

