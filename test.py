import datetime
import os
import re
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.utils.data
from torch import nn
from einops import rearrange

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F

import torch.distributed as dist
import util.misc as _utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data.a2d_eval import calculate_precision_at_k_and_iou_metrics
import pycocotools.mask as mask_util


def get_dataset(image_set, transform, args):
    if args.dataset == 'refcoco' or args.dataset == 'refcoco+' or args.dataset == 'refcocog':
        from data.dataset_refer_bert import ReferDataset
        ds = ReferDataset(args,
                          split=image_set,
                          image_transforms=transform,
                          target_transforms=None,
                          eval_mode=True
                          )
    elif args.dataset == 'a2d':
        from data.a2d import build_a2d_dataset
        ds = build_a2d_dataset(args, image_set)

    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I * 1.0 / U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def batch_iou(pred_seg, gd_seg):
    # pred_seg shape: (b, 1, h, w) or (b, h, w)
    # gd_seg shape: (b, 1, h, w) or (b, h, w)
    I = np.sum(np.logical_and(pred_seg, gd_seg), axis=(-2, -1), keepdims=False)  # (b, 1) or (b)
    U = np.sum(np.logical_or(pred_seg, gd_seg), axis=(-2, -1), keepdims=False)  # (b, 1) or (b)

    return I, U


def evaluate_a2d(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            masks = target['masks']  # [B, T, H, W]
            valid_indices = target['valid_indices']
            image, masks, sentences, attentions = image.to(device), masks.to(device), sentences.to(
                device), attentions.to(device)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            masks = rearrange(masks, 'b t h w -> (b t) h w')  # [B, H, W] if T = 1, else (B*T, H, W)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                if args.save_feats:
                    output, bag_feats = model.forward_feats(image, sentences, l_mask=attentions)
                    # colormap = cm.RdBu_r
                    # 4 feature maps
                    to_store = []
                    t = image.size(1)
                    for feats in bag_feats:
                        valid_indices = torch.tensor([i * t + ind for i, ind in enumerate(valid_indices)]).to(device)
                        feats = torch.index_select(feats, 0, valid_indices)
                        to_store.append(feats.squeeze(0).cpu().numpy())

                    image_id = target['image_id'][0]
                    matchObj = re.match(r'v_(.*)_f_(.*)_i_(.*)', image_id, re.M | re.I)
                    if matchObj:
                        video_id = matchObj.group(1)
                        frame_id = matchObj.group(2)
                        instance_id = matchObj.group(3)
                    sub_folder_name = os.path.join('./a2d_visualized_feats', video_id, frame_id)

                    if not os.path.isdir(sub_folder_name):
                        os.makedirs(sub_folder_name)
                    for a in range(4, 0, -1):
                        # residual_Cs[a].save(os.path.join(path_to_save, 'residual_C' + str(a+1) + '.jpg'))
                        # default: 'viridis'
                        plt.imsave(os.path.join(sub_folder_name, instance_id + '_L' + str(a) + '.png'), to_store[4 - a],
                                   cmap='RdBu_r')
                else:
                    output = model(image, sentences, l_mask=attentions)
                t = image.size(1)
                valid_indices = torch.tensor([i * t + ind for i, ind in enumerate(valid_indices)]).to(device)
                output_valid = torch.index_select(output, 0, valid_indices)

            if args.a2d_masks:
                output_valid = output_valid.argmax(1)  # (B*T, H, W)
                output_valid = output_valid.squeeze()  # (H, W) (480, 480)
                output_valid = output_valid.cpu().numpy().astype(np.uint8)  # (orig_h, orig_w), np.uint8
                output_valid = Image.fromarray(output_valid)

                image_id = target['image_id'][0]
                matchObj = re.match(r'v_(.*)_f_(.*)_i_(.*)', image_id, re.M | re.I)
                if matchObj:
                    video_id = matchObj.group(1)
                    frame_id = matchObj.group(2)
                    instance_id = matchObj.group(3)
                path = os.path.join('./a2d_predicted_masks', video_id, frame_id)

                if not os.path.isdir(path):
                    os.makedirs(path)
                output_valid.save(path + '/' + instance_id + '.png')
                continue

            iou, I, U = IoU(output_valid, masks)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

    if args.a2d_masks:
        return

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)  # (B*T, H, W)

    intersection = torch.sum(torch.mul(pred, gt))
    # scalar tensor; if B*T != 1, then this does not strictly lead to mean IoU
    union = torch.sum(torch.add(pred, gt)) - intersection  # scalar tensor

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def main(args):
    integrated = (args.model == 'lavt_one' or args.model == 'lavt_video' or args.model == 'lts' or args.model == 'vlt')
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    if args.ckpt == True:
        single_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    if not integrated:
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    if args.dataset == "a2d":
        evaluate_a2d(model, data_loader_test, bert_model, device=device)
    else:
        evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)

