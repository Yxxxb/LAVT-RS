import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('--att_norm_layer_type', default='IN', help='type of norms used in f_query and f_w.'
                                                                    'IN, InstanceNorm1D, is the default,'
                                                                    'BN: BatchNorm1D, LN: LayerNorm,'
                                                                    'none: Identity.')
    parser.add_argument('--a2d_data_root', default='./data/A2D/Release/', help='the A2D dataset root directory')
    parser.add_argument('--a2d_masks', action='store_true',
                        help='If set, output masks of a2d while testing.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bcam', action='store_true', help='if true, use BCAM as the attention module')
    parser.add_argument('--bert_tokenizer',  default='bert-base-uncased', help='BERT tokenizer')

    parser.add_argument('--cat_reduce_3', action='store_true', help='in TS-PWAM, use 2d conv 3x3 to reduce'
                                                                    'concatenated TPWAM and SPWAM features.')
    parser.add_argument('--ck_bert',  default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--ckpt', action='store_true', help='if true, finetune with checkpoint models.')
    parser.add_argument('--clip_length', default=8, type=int)
    parser.add_argument('--conv3d_kernel_size', default='3-1-1',
                        help='If specified, should be in the format of '
                              'a-b-c, e.g., 3-3-3,'
                              'where a, b, and c refer to the kerner sizes of d, h, and w dims')
    parser.add_argument('--conv3d_kernel_size_t', default='3-1-1',
                        help='If specified, should be in the format of '
                              'a-b-c, e.g., 3-3-3,'
                              'where a, b, and c refer to the kerner sizes of d, h, and w dims')
    parser.add_argument('--conv3d_kernel_size_s', default='1-1-1',
                        help='If specified, should be in the format of '
                              'a-b-c, e.g., 1-1-1,'
                              'where a, b, and c refer to the kerner sizes of d, h, and w dims')
    parser.add_argument('--conv3d_kernel_size_sq', default='1-3-3',
                        help='If specified, should be in the format of '
                              'a-b-c, e.g., 1-1-1,'
                              'where a, b, and c refer to the kerner sizes of d, h, and w dims')
    
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, refcocog, a2d, ytvos, joint'
                                                             'refs+ytvos')
    parser.add_argument('--davis_data_root', default='./data/DAVIS/', help='the DAVIS dataset root directory')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--debug', action='store_true', help='If true, debug pycocotools precision and recall.')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--efn', action='store_true', help='If true, use EFN naive attention for fusion; internally, we call this model EFN;'
                                                           'full EFN has a more complicated attention and a Spatial Transformer network.')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fix_lr', action='store_true', help='If true, fix --lr during training; This is used'
                                                              'in experimental fine-tuning.')
    parser.add_argument('--fuse', default='default',
                        help='default: use PWAM (pixel-word attention and element-wise multiplication)'
                             'simple: use average pooled sentence features and element-wise multiplication')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--gacd', action='store_true', help='if true, use GA (GARAN) as the attention module')
    parser.add_argument('--hs', action='store_true',
                        help='If true, use sum of visual and mm features for prediction'
                             'otherwise, we only use mm features for prediction'
                             'Implementation-wise, the True/False difference is to return x or x_residual')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')

    parser.add_argument('--image_combined_3d_pretrain', action='store_true',
                        help='if true, train a 3d conv LAVT model on the combined three RefCOCO datasets.')
    
    parser.add_argument('--interpolate_before_seg', action='store_true',
                        help='if true, interpolate to the size of H/2 before segmantation.')

    parser.add_argument('--lang_enc_params', default='encoder-10', help='encoder-10, encoder-all,'
                                                                        'embeddings+encoder-10'
                                                                        'embeddings+encoder-all')
    parser.add_argument('--lazy_pred', action='store_true',
                        help='If true, use the output from a swin layer for prediction'
                             'and not the input or mm features'
                             'this is a standalone option that is orthogonal to --version and --fuse;'
                             'but mutually exclusive with --hs')
    parser.add_argument('--lg_act_layer', default='tanh', help='the last activation layer in LG; tanh or sigmoid')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--loss', default='ce', help='ce, mc_dice, dice_boundary or dice_focal')
    parser.add_argument('--loss_focal_rate', default=3, type=int, help='focal loss rate')
    parser.add_argument('--loss_boundary_rate', default=0.05, type=float, help='boundary loss rate')
    parser.add_argument('--loss_dice_rate', default=1, type=float, help='dice loss rate')

    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--lr_upsample', default=0.00003, type=float, help='the learning rate used in fine-tune a2d upsample module')
    parser.add_argument('--map_score', default='mask_pool',
                        help='Choose from [mask_pool, iou]. if mask_pool, use argmax to obtain 0/1 mask for the object,'
                              'then use the mask to pool object class score map (after softmax). If IoU, compute IoU'
                              'and use this iou as the confidence score for coco mAP, AP, and AR evaluation.')
    
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--mm_3', action='store_true',
                        help='If set, turn project_mm into 3x3 2d conv; currently only implemented for SEP-T-PWAM')
    parser.add_argument('--mm_3x3', action='store_true',
                        help='If set, turn project_mm into 3x3x3 3d conv; currently only implemented for SEP-T-PWAM'
                             'and TS-PWAM.')
    parser.add_argument('--mm_t3x3_s1x1', action='store_true',
                        help='If set, use two project_mm, one 3x3x3 3d conv, the other 1x1x1 3d conv;'
                             'currently only implemented for SEP-T-PWAM and TS-PWAM.')
    parser.add_argument('--model', default='lavt', help='model: lavt, lavt_one, lavt_video')
    parser.add_argument('--model_id', default='lavt', help='name to identify the model')
    parser.add_argument('--ngpus', default=1, type=int, help='number of GPUs used for inference on YTVOS only.')
    parser.add_argument('--not_consecutive', action='store_true',
                        help='if set, do not use consecutive frames as the clip input, but use randomly samples frames,'
                             ' as we did for training; only has effect for the inference on A2D. ')
    parser.add_argument('--num_frames', default=1, type=int, help='number of frames per input video clip')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--pretrained2d_lavt_weights', default='',
                        help='if non-empty, then this is the path to weights of an'
                             'image LAVT model pre-trained on image datasets. Note that this is NOT'
                             'pretrained 2D Swin weights; this should be LAVT weights (including the PWAM and'
                             'LG and classifier weights). The relevant function is written in _utils.py')
    parser.add_argument('--pretrained2d_lavt_weights_for_a_3d_model', default='',
                        help='Same as the above but the loading target is a 3D conv LAVT; '
                             'if non-empty, then this is the path to weights of an'
                             'image LAVT model pre-trained on image datasets. Note that this is NOT'
                             'pretrained 2D Swin weights; this should be LAVT weights (including the PWAM and'
                             'LG and classifier weights). The relevant function is written in _utils.py')
    parser.add_argument('--pretrained_video_lavt_weights_on_refcocos', default='',
                        help='The path to pre-trained video LAVT (with Video Swin and 3D PWAMs '
                             'as described in the manuscript) weights, pre-trained on the three RefCOCO/+/g datasets'
                             'concatenated training sets.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--pseudo_video_aug', default='',
                        help='vanilla: simply duplicate the image num_frames times to form a clip;'
                             'weak-rotate-translate-shear-crop: experimental, these needs to be gentle because'
                             'strong augmentations are likely to destroy critical location information for identifying'
                             'the target without ambiguity, and also, cropping might eliminate distractors, making'
                             'learning less challenging.')
    parser.add_argument('--ref_image_combined_pretrain', action='store_true',
                        help='If true, train image LAVT on three combined RefCOCO/+/g datasets.')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--res', action='store_true', help='Only relevant if --seq_t_pwam or --sep_seq_t_pwam'
                                                           'is specified.'
                                                           'If true, use P3D-C variant with a residual connection')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--sample_3', action='store_true',
                        help='only relevant to the JHMDB dataset. If true, uniformly sample 3 frames per video for'
                             'inference.')
    parser.add_argument('--save_feats', action='store_true',
                        help='Save model features.')
    parser.add_argument('--seg_last', action='store_true',
                        help='Segmentation after all upsampling operations.')
    parser.add_argument('--sep_seq_t_pwam', action='store_true',
                        help='variant of --sep_t_pwam, but within the temporal branch, we decouple 3x3x3 into 3x1x1 and'
                             '1x3x3')
    parser.add_argument('--sep_seq_t_pwam_inner', action='store_true',
                        help='sep_seq_t_pwam is a variant of --sep_t_pwam,'
                             'but within the temporal branch, we decouple 3x3x3 into 3x1x1 and 1x3x3;'
                             'this INNER version means that apply --sep_seq_t_pwam only to the inner visual query'
                             'projection for the attentional step and NOT on the outside visual feature projection'
                             'used for element-wise multiplication.')
    parser.add_argument('--sep_t_pwam', action='store_true', help='if true, use what hs and ys said to fuse s and t via addition'
                                                                  'features within pwam rather than using two pwams.')
    parser.add_argument('--sep_t_pwam_inner', action='store_true',
                        help='if true, use what hs and ys said to fuse s and t via addition'
                             'features within pwam rather than using two pwams.'
                             'BUT, apply this change only to the inner visual query projection for the attentional'
                             'step and NOT on the outside visual feature projection used for element-wise'
                             'multiplication.')
    parser.add_argument('--sept_sum_3_kernel_size', default='',
                        help='If specified, should be in the format of '
                        'a-b-c, e.g., 1-1-1,'
                        'where a, b, and c refer to the kernel sizes of d, h, and w dims')
    parser.add_argument('--sept_cat_reduce_kernel_size', default='',
                        help='If specified, should be in the format of '
                        'a-b-c, e.g., 1-1-1,'
                        'where a, b, and c refer to the kernel sizes of d, h, and w dims')
    parser.add_argument('--seq_t_pwam', action='store_true', help='if true, use a spatial 3d conv first followed by a'
                                                                  'temporal 3d conv. P3D-A. if --res is specified,'
                                                                  'then would be the P3D-C variant')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--s_tanh_plus_1_gate_1_q', action='store_true', help='if true, apply a tanh + 1 self attention'
                                                                            'gate on 111 features in the query'
                                                                              'in sep-t-pwam;'
                                                                            '(before the addition and '
                                                                            'similarity matmul).')
    parser.add_argument('--s_tanh_plus_1_gate_1_v', action='store_true', help='if true, apply a tanh + 1 self attention'
                                                                            'gate on 111 features in outer visual feats'
                                                                              'in sep-t-pwam;'
                                                                            '(before the addition and '
                                                                            'similarity matmul).')
    parser.add_argument('--t_tanh_plus_1_gate_1_q', action='store_true', help='if true, apply a tanh + 1 self attention'
                                                                            'gate on 333 features in the query '
                                                                              'in sep-t-pwam;'
                                                                            '(before the addition and '
                                                                            'similarity matmul).')
    parser.add_argument('--t_tanh_plus_1_gate_1_v', action='store_true', help='if true, apply a tanh + 1 self attention'
                                                                            'gate on 333 features in outer visual feats '
                                                                              'in sep-t-pwam;'
                                                                            '(before the addition and '
                                                                            'similarity matmul).')
    parser.add_argument('--ts_pwam', action='store_true', help='if true, use two pwams, one original, one'
                                                               'with Conv3d for processing videos.')
    parser.add_argument('--tspwam_sum', action='store_true', help='if true, in TSPWAM, use sum rather than'
                                                                  'concatenation at the end for combining'
                                                                  'the two types of mm feats (spatial & temporal)')
    parser.add_argument('--t_pwam', action='store_true', help='if true, use one naive 3D conv pwam,'
                                                              'for processing videos.')
    parser.add_argument('--t_pwam_comp', action='store_true', help='if true, use one naive 3D conv pwam,'
                                                                   'for processing videos.'
                                                                   'This uses 3D convs for all 4 spatial '
                                                                   'transforms.')
    parser.add_argument('--test_fake_method', default='add_first', help='test_fake_method: add_first, add_later')
    parser.add_argument('--use_amp', action='store_true',
                        help='If set, use automatic mixed precision for training. Only training on YTVOS needs this'
                             'because this dataset is too big.')
    parser.add_argument('--use_checkpoint', action='store_true', help='Swin\'s use_checkpoint argument')
    parser.add_argument('--version', default='default',
                        help='default: tanh gate with 0 init to feed back'
                             'no_gate: add back multi-modal features without applying a gate function'
                             'none: do not add back multi-modal features')

    # Full model: --version default --fuse default, or don't specify any of these
    # PWAM-only: --version none --fuse default or don't specify --fuse
    # LP-only: --version default --fuse simple
    # super baseline: --version none --fuse simple

    parser.add_argument('--visualize', action='store_true',
                        help='If set, save mask visualizations of YTVOS evaluation (only used when testing on YTVOS).')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='before: only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.'
                             'now with video swin: specify it if we are finetuning a window 12 model pre-trained'
                             'on image datasets.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--w_3', action='store_true',
                        help='If set, turn W into 3x3 2D conv; currently only implemented for SEP-T-PWAM.')
    parser.add_argument('--w_3x3', action='store_true',
                        help='If set, turn W into 3x3x3 3D conv; currently only implemented for SEP-T-PWAM and TS-PWAM.')
    parser.add_argument('--w_t3x3_s1x1', action='store_true',
                        help='If set, use two Ws, one 3x3x3 3D conv, the other 1x1x1 3D conv;'
                             'currently only implemented for SEP-T-PWAM and TS-PWAM.')
    parser.add_argument('--ytvos_data_root', default='./data/ReferringYouTubeVOS2021/',
                        help='the referring YouTube-VOS dataset root directory')
    parser.add_argument('--ytvos_2d_swin_pwam', action='store_true',
                        help='If set, use 2D-Swin-Trasnformer as the backbone network and train it with PWAM on the video dataset YouTube-VOS.')
    parser.add_argument('--ytvos_2d_swin_3d_pwam', action='store_true',
                        help='If set, use 2D-Swin-Trasnformer as the backbone network and train it with 3D-PWAM on the video dataset YouTube-VOS.')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
