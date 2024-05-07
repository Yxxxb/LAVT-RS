
from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None
        self.lazy_pred = args.lazy_pred

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        '''
        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
        '''
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        if not self.lazy_pred:
            x_c1, x_c2, x_c3, x_c4 = features
        else:
            x_c1 = None
            x_c2, x_c3, x_c4 = features  # each has shape: (B, Ci, Hi, Wi)
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass


#########################################################################
# LAVT Video: use Video Swin Transformer as the vision backbone network #
# (uses the above lavt_one implementation)
# the only difference is a transpose on the input images'
# channel dim and temporal dim
#########################################################################
class _LAVTVideoSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTVideoSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None
        self.lazy_pred = args.lazy_pred
        self.seg_last = args.seg_last

    def forward(self, x, text, l_mask):
        # input x shape: (B, T, 3, H, W)
        # input text shape: (B, 22)
        # input l_mask shape: (B, 22)
        input_shape = x.shape[-2:]  # (H, W)
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (b, 22, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        # video swin expects an input size of (B, 3, T, H, W), so we need to transpose the 3 and T dims of x
        x = x.permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        features = self.backbone(x, l_feats, l_mask)  # 4-tuple each with shape: (B*T, Ci, Hi, Wi)
        if not self.lazy_pred:
            x_c1, x_c2, x_c3, x_c4 = features  # each has shape: (B*T, Ci, Hi, Wi)
        else:
            x_c1 = None
            x_c2, x_c3, x_c4 = features  # each has shape: (B*T, Ci, Hi, Wi)
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)  # (B*T, 2, H4, W4)
        if not self.seg_last:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)  # (B*T, 2, H, W)

        return x

    def forward_feats(self, x, text, l_mask):
        # input x shape: (B, T, 3, H, W)
        # input text shape: (B, 22)
        # input l_mask shape: (B, 22)
        input_shape = x.shape[-2:]  # (H, W)
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (b, 22, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        # video swin expects an input size of (B, 3, T, H, W), so we need to transpose the 3 and T dims of x
        x = x.permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        features = self.backbone(x, l_feats, l_mask)  # 4-tuple each with shape: (B*T, Ci, Hi, Wi)
        if not self.lazy_pred:
            x_c1, x_c2, x_c3, x_c4 = features  # each has shape: (B*T, Ci, Hi, Wi)
        else:
            x_c1 = None
            x_c2, x_c3, x_c4 = features  # each has shape: (B*T, Ci, Hi, Wi)
        x, feats = self.classifier.forward_feats(x_c4, x_c3, x_c2, x_c1)  # (B*T, 2, H4, W4)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)  # (B*T, 2, H, W)

        return x, feats

    def load_from_pretrained2d_lavt_weights(self, pretrained):
        # pretrained is weights path
        # train is whether we are loading these weights for further fine-tuning or for testing

        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # note: relative_position_index is a registered buffer
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it (note: but I don't know we save attn_mask in our model?????)
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        assert 'backbone.patch_embed.proj.weight' in state_dict
        # the patch temporal length in our segmentation model is always 1
        state_dict['backbone.patch_embed.proj.weight'] = state_dict['backbone.patch_embed.proj.weight'].unsqueeze(2)

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.backbone.window_size[1] - 1) * (2 * self.backbone.window_size[2] - 1)
            wd = self.backbone.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.backbone.window_size[1] - 1, 2 * self.backbone.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)  # we can't load the weights strictly, since we always
        # delete relative_position_index in source dict
        print(msg)
        print(f"=> loaded successfully '{pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def load_from_pretrained2d_lavt_weights_into_a_3d_model(self, pretrained):
        # pretrained is weights path
        # train is whether we are loading these weights for further fine-tuning or for testing

        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # note: relative_position_index is a registered buffer
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it (note: but I don't know we save attn_mask in our model?????)
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # delete fusion because it is different in 2d lavt and in 3d lavt
        fusion_keys = [k for k in state_dict.keys() if ".fusion" in k]
        for k in fusion_keys:
            del state_dict[k]

        assert 'backbone.patch_embed.proj.weight' in state_dict
        # the patch temporal length in our segmentation model is always 1
        state_dict['backbone.patch_embed.proj.weight'] = state_dict['backbone.patch_embed.proj.weight'].unsqueeze(2)

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.backbone.window_size[1] - 1) * (2 * self.backbone.window_size[2] - 1)
            wd = self.backbone.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.backbone.window_size[1] - 1, 2 * self.backbone.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)  # we can't load the weights strictly, since we always
        # delete relative_position_index in source dict
        print(msg)
        print(f"=> loaded successfully '{pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()


class LAVTVideo(_LAVTVideoSimpleDecode):
    pass


#########################################################################
# Swin-LTS:
#########################################################################
class _LTS(nn.Module):
    def __init__(self, backbone, classifier, args=None):
        super(_LTS, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (b, 22, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x)
        x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, l_feats, l_mask)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LTS(_LTS):
    pass


#########################################################################
# Swin-VLT:
#########################################################################
class _VLT(nn.Module):
    def __init__(self, backbone, classifier, args=None, link=None):
        super(_VLT, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.link = link
        self.model = args.model
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (b, 22, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        # if self.model == 'vlt_swin_transformer' or self.model == 'vlt_resnet':
        features = self.backbone(x)
        x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, l_feats, l_mask)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class VLT(_VLT):
    pass



#########################################################################
# LAVT-VLT:
#########################################################################
class _LAVT_VLT(nn.Module):
    def __init__(self, backbone, classifier, args=None, link=None):
        super(_LAVT_VLT, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.link = link
        self.model = args.model
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (b, 22, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        # if self.model == 'vlt_swin_transformer' or self.model == 'vlt_resnet':
        features = self.backbone(x, l_feats, l_mask)
        x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, l_feats, l_mask)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT_VLT(_LAVT_VLT):
    pass

