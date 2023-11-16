import torch
import torch.nn as nn
from .mask_predictor import SimpleDecoding, LTSDecoding
from .backbone import MultiModalSwinTransformer, SwinTransformer
from .video_swin_transformer import MultiModalSwinTransformer3D
from ._utils import LAVT, LAVTOne, LAVTVideo, LTS, VLT, LAVT_VLT
from .vlt import VLTFuseAndClassify


__all__ = ['lavt', 'lavt_one', 'lavt_video', 'lts', 'vlt', 'lavt_vlt']


# LAVT
def _segm_lavt(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVT]

    classifier = model_map[0](8*embed_dim, args)
    base_model = model_map[1]

    model = base_model(backbone, classifier)
    return model


def _load_model_lavt(pretrained, args):
    model = _segm_lavt(pretrained, args)
    return model


def lavt(pretrained='', args=None):
    return _load_model_lavt(pretrained, args)


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
def _segm_lavt_one(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    if args.lazy_pred:
        out_indices = (1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop, args=args
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVTOne]

    classifier = model_map[0](8*embed_dim, args)
    base_model = model_map[1]

    model = base_model(backbone, classifier, args)
    return model


def _load_model_lavt_one(pretrained, args):
    model = _segm_lavt_one(pretrained, args)
    return model


def lavt_one(pretrained='', args=None):
    return _load_model_lavt_one(pretrained, args)


#############################################################
# LAVT Video: use Video Swin as the vision backbone network #
#############################################################
def _segm_lavt_video(pretrained, args):
    # initialize the Video SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        drop_path_rate = 0.1
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
        drop_path_rate = 0.2
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
        drop_path_rate = 0.3
    else:
        assert False
    patch_size = (1, 4, 4)
    if not args.window12:
        window_size = (8, 7, 7)
    else:
        window_size = (8, 12, 12)
    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]
    out_indices = (0, 1, 2, 3)
    if args.lazy_pred:
        out_indices = (1, 2, 3)

    backbone = MultiModalSwinTransformer3D(patch_size=patch_size, embed_dim=embed_dim,
                                           depths=depths, num_heads=num_heads,
                                           window_size=window_size, drop_path_rate=drop_path_rate,
                                           patch_norm=True, out_indices=out_indices,
                                           use_checkpoint=args.use_checkpoint, num_heads_fusion=mha,
                                           fusion_drop=args.fusion_drop, args=args
                                           )
    # proj in PatchEmbed3D is a Conv3D, its kernel_size is patch_size;
    # in the above swin initialization we have already specified patch_size as
    # (1, 4, 4); hence when loading pre-trained weights for training we only need to change the shape of the weights
    # and don't need to change the shape of the proj function in PatchEmbed3D
    # This practice is followed in init_weights(). See the function for details.
    if pretrained:
        print('Initializing Video Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Video Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    model_map = [SimpleDecoding, LAVTVideo]  

    classifier = model_map[0](8*embed_dim, args) 
    base_model = model_map[1]

    model = base_model(backbone, classifier, args) 
    return model


def _load_model_lavt_video(pretrained, args):
    model = _segm_lavt_video(pretrained, args)
    return model


def lavt_video(pretrained='', args=None):
    return _load_model_lavt_video(pretrained, args)


##########################################
# Reference methods: LTS, EFN, VLT, etc. #
##########################################
# Swin-LTS #
############
def _segm_lts(pretrained, args):
    # initialize SwinTransformer backbone with corresponding configurations
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        print('Window size 7!')
        window_size = 7

    out_indices = (1, 2, 3)  # LTS only needs the last three layers of outputs

    backbone = SwinTransformer(embed_dim=embed_dim,
                               depths=depths,
                               num_heads=num_heads,
                               window_size=window_size,
                               ape=False,
                               drop_path_rate=0.3,
                               patch_norm=True,
                               out_indices=out_indices,
                               use_checkpoint=False
                               )

    if pretrained:
        print('Initializing pre-trained Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Swin Transformer weights.')
        backbone.init_weights()

    model_map = [LTSDecoding, LTS]

    classifier = model_map[0](8*embed_dim, args)
    base_model = model_map[1]

    model = base_model(backbone, classifier, args=args)
    return model


def _load_model_lts(pretrained, args):
    model = _segm_lts(pretrained, args)
    return model


def lts(pretrained='', args=None):
    print('Running LTS!!!!!!!!!!!')
    return _load_model_lts(pretrained, args)


##############
## Swin-VLT ##
##############
def _vlt(pretrained, args):
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False

    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        print('Window size 7!')
        window_size = 7

    out_indices = (1, 2, 3)  # VLT uses last 3 outputs

    backbone = SwinTransformer(embed_dim=embed_dim,
                               depths=depths,
                               num_heads=num_heads,
                               window_size=window_size,
                               ape=False,
                               drop_path_rate=0.3,
                               patch_norm=True,
                               out_indices=out_indices,
                               use_checkpoint=False
                               )

    if pretrained:
        print('Initializing pre-trained Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize VLT Swin Transformer weights.')
        backbone.init_weights()

    model_map = [VLTFuseAndClassify, VLT]
    classifier = model_map[0](args=args)
    # d_model=256, nhead=8, d_hid=256, nlayers=2, args=None
    model = model_map[1](backbone, classifier, args=args)
    # backbone, classifier, args=None
    return model


def _load_model_vlt(pretrained, args):
    model = _vlt(pretrained, args)
    return model


def vlt(pretrained='', args=None):
    print('Running VLT with Swin backbone!!!!!!!')
    return _load_model_vlt(pretrained, args)


######################################
## LAVT as encoder + VLT as decoder ##
######################################
def _lavt_vlt(pretrained, args):
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False

    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        print('Window size 7!')
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (1, 2, 3)  # VLT uses last 3 outputs
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop, args=args
                                         )
    if pretrained:
        print('Initializing pre-trained Swin Transformer backbone from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize LAVT + VLT weights.')
        backbone.init_weights()

    model_map = [VLTFuseAndClassify, LAVT_VLT]
    classifier = model_map[0](args=args)
    # d_model=256, nhead=8, d_hid=256, nlayers=2, args=None
    model = model_map[1](backbone, classifier, args=args)
    # backbone, classifier, args=None
    return model


def _load_model_lavt_vlt(pretrained, args):
    model = _lavt_vlt(pretrained, args)
    return model


def lavt_vlt(pretrained='', args=None):
    print('Running VLT with LAVT as the backbone!!!!!!!')
    return _load_model_lavt_vlt(pretrained, args)

