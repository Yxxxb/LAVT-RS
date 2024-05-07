import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from einops import rearrange

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation

import transforms as T
import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict

from losses import MultiClassDiceLoss, cross_entropy_loss, DiceFocalLoss, DiceBoundaryLoss


def get_dataset(image_set, transform, args):
    if args.dataset == 'refcoco' or args.dataset == 'refcoco+' or args.dataset == 'refcocog':
        from data.dataset_refer_bert import ReferDataset
        ds = ReferDataset(args,
                          split=image_set,
                          image_transforms=transform,
                          target_transforms=None
                          )
    elif args.dataset == 'a2d':
        from data.a2d import build_a2d_dataset
        ds = build_a2d_dataset(args, image_set)
    elif args.dataset == 'ytvos':
        from data.ytvos import build_ytvos_dataset
        ds = build_ytvos_dataset(args)
    elif args.dataset == 'joint':
        from data.concat_dataset import build_joint_dataset
        ds = build_joint_dataset(image_set, args)
    else:
        assert False

    num_classes = 2

    return ds, num_classes


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


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

def evaluate(model, data_loader, bert_model):
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
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

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


def evaluate_ref_3d(model, data_loader):
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
            images, targets = data
            images = images.cuda(non_blocking=True)  # [B, T, 3, H, W]
            masks = targets['masks']  # [B, T, H, W]
            sentences = targets['caption']  # [B, 1, 22]
            attentions = targets['caption_mask']  # [B, 1, 22]
            masks, sentences, attentions = masks.cuda(non_blocking=True), \
                                           sentences.cuda(non_blocking=True), attentions.cuda(non_blocking=True)

            # images = images.squeeze(1)  # [B, 3, H, W] if T = 1, else [B, T, 3, H, W]
            # masks = masks.squeeze(1)  # [B, H, W] if T = 1, else [B, T, H, W]
            masks = rearrange(masks, 'b t h w -> (b t) h w')  # [B, H, W] if T = 1, else (B*T, H, W)
            sentences = sentences.squeeze(1)  # (B, 22)
            attentions = attentions.squeeze(1)  # (B, 22)

            output = model(images, sentences, l_mask=attentions)  # (B*T, 2, H, W)
            #loss = criterion(output, masks)  # scalar tensor, needs item()

            ## compute ious ##
            iou, I, U = IoU(output, masks)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1
        iou = acc_ious / total_its

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


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)
        # image [B, C, H, W]
        sentences = sentences.squeeze(1)  # [B, 20]
        attentions = attentions.squeeze(1)  # [B, 20]
        
        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, l_mask=attentions)
        else:
            output = model(image, sentences, l_mask=attentions)  # output [B, 2, H, W]

        loss = criterion(output, target)
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()

        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_one_epoch_a2d(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                           iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    acc_ious = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        masks = target['masks']  # [B, T, H, W]
        valid_indices = target['valid_indices']
        image, masks = image.cuda(non_blocking=True), \
                                       masks.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        masks = rearrange(masks, 'b t h w -> (b t) h w')  # [B, H, W] if T = 1, else (B*T, H, W)
        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (B, 22, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, l_mask=attentions)
            loss = criterion(output, masks)  # scalar tensor, needs item() if logging
        else:
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                output = model(image, sentences, l_mask=attentions)  # (B*T, 2, H, W)
                t = image.size(1)
                valid_indices = torch.tensor([i * t + ind for i, ind in enumerate(valid_indices)]).cuda()
                output_valid = torch.index_select(output, 0, valid_indices)
                loss = criterion(output_valid, masks)  # scalar tensor, needs item()

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        ## compute ious ##
        iou, I, U = IoU(output_valid, masks)  # scalar tensors; not strictly per-frame IoU; is mini-batch*T-on-this-GPU IoU
        acc_ious += iou
        mean_IoU.append(iou)
        cum_I += I
        cum_U += U
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
        seg_total += 1

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data, output_valid, valid_indices
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # at the end of epoch, summarize and return some stats; stats are not synced across cards, only master got printed
    iou = acc_ious / total_its
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)  # the same as iou; a verification
    print('Master process curr epoch training results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def evaluate_a2d(model, data_loader, bert_model):
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
            image, masks = image.cuda(non_blocking=True), masks.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            masks = rearrange(masks, 'b t h w -> (b t) h w')  # [B, H, W] if T = 1, else (B*T, H, W)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)
                t = image.size(1)
                valid_indices = torch.tensor([i * t + ind for i, ind in enumerate(valid_indices)]).cuda()
                output_valid = torch.index_select(output, 0, valid_indices)

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

    precision_K = []
    for n_eval_iou in range(len(eval_seg_iou_list)):
        precision_K.append(seg_correct[n_eval_iou] * 100. / seg_total)

    return 100 * iou, 100 * cum_I / cum_U, precision_K[0], precision_K[1], precision_K[2], precision_K[3], precision_K[4]


def train_one_epoch_ytvos(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                          iterations, bert_model, args, scaler):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0
    acc_ious = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        images, targets = data
        images = images.cuda(non_blocking=True)  # [B, T, 3, H, W]

        masks = targets['masks']  # [B, T, H, W]
        sentences = targets['caption']  # [B, 1, 22]
        attentions = targets['caption_mask']  # [B, 1, 22]
        masks, sentences, attentions = masks.cuda(non_blocking=True), \
                                       sentences.cuda(non_blocking=True), attentions.cuda(non_blocking=True)

        if not args.image_combined_3d_pretrain:
            images = images.squeeze(1)  # [B, 3, H, W] if T = 1, else [B, T, 3, H, W]
        # masks = masks.squeeze(1)  # [B, H, W] if T = 1, else [B, T, H, W]
        masks = rearrange(masks, 'b t h w -> (b t) h w')  # [B, H, W] if T = 1, else (B*T, H, W)
        sentences = sentences.squeeze(1)  # (B, 22)
        attentions = attentions.squeeze(1)  # (B, 22)

        if args.ytvos_2d_swin_pwam:
            images = rearrange(images, 'b t c h w -> (b t) c h w')  # [B*T, C, H, W], 2D-SwinT requires images input with [B, C, H, W].
            B, _ = sentences.shape
            T = args.num_frames
            sentences = sentences.unsqueeze(dim=1)  # [B, 1, 22]
            sentences = sentences.expand(-1, T, -1)
            sentences = sentences.reshape(B * T, 22)  # [B*T, 22]
            attentions = attentions.unsqueeze(dim=1)  # [B, 1, 22]
            attentions = attentions.expand(-1, T, -1)
            attentions = attentions.reshape(B * T, 22)  # [B*T, 22]

        elif args.ytvos_2d_swin_3d_pwam:
            images = rearrange(images, 'b t c h w -> (b t) c h w')  # [B*T, C, H, W], 2D-SwinT requires images input with [B, C, H, W].

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (B, 22, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(images, embedding, l_mask=attentions)
            loss = criterion(output, masks)  # scalar tensor, needs item() if logging
        else:
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                output = model(images, sentences, l_mask=attentions)  # (B*T, 2, H, W)
                loss = criterion(output, masks)  # scalar tensor, needs item()

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ##############################
        lr_scheduler.step()

        ## compute ious ##
        iou, I, U = IoU(output, masks)  # scalar tensors; not strictly per-frame IoU; is mini-batch*T-on-this-GPU IoU
        acc_ious += iou
        mean_IoU.append(iou)
        cum_I += I
        cum_U += U
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
        seg_total += 1

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"], iou=iou)

        del images, masks, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # at the end of epoch, summarize and return some stats; stats are not synced across cards, only master got printed
    iou = acc_ious / total_its
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)  # the same as iou; a verification
    print('Master process curr epoch training results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def main(args):
    integrated = (args.model == 'lavt_one' or args.model == 'lavt_video' or args.model == 'lts' or args.model == 'vlt')
    # datasets
    if args.ref_image_combined_pretrain:
        # the dataset and splitBy values passed via command line are no longer relevant because we re-define them here
        dataset_list = []
        test_dataset_list = []
        for dataset_name, split_by in zip(['refcoco', 'refcoco+', 'refcocog'], ['unc', 'unc', 'umd']):
            args.dataset = dataset_name
            args.splitBy = split_by
            dataset_list.append(get_dataset("train", get_transform(args=args), args=args)[0])
            test_dataset_list.append(get_dataset("val", get_transform(args=args), args=args)[0])
        dataset = torch.utils.data.ConcatDataset(dataset_list)
        dataset_test = torch.utils.data.ConcatDataset(test_dataset_list)
    elif args.image_combined_3d_pretrain:
        from data.refer_video import build_refpseudovideo_dataset
        dataset_list = []
        test_dataset_list = []
        # three refcoco datasets
        for dataset_name, split_by in zip(['refcoco', 'refcoco+', 'refcocog'], ['unc', 'unc', 'umd']):
            args.dataset = dataset_name
            args.splitBy = split_by
            dataset_list.append(build_refpseudovideo_dataset('train', args))
            test_dataset_list.append(build_refpseudovideo_dataset('val', args))
        # NO ytv dataset
        dataset = torch.utils.data.ConcatDataset(dataset_list)
        dataset_test = torch.utils.data.ConcatDataset(test_dataset_list)
    else:
        if args.dataset == 'refs+ytvos':
            from data.refer_video import build_refpseudovideo_dataset
            dataset_list = []
            # three refcoco datasets
            for dataset_name, split_by in zip(['refcoco', 'refcoco+', 'refcocog'], ['unc', 'unc', 'umd']):
                args.dataset = dataset_name
                args.splitBy = split_by
                dataset_list.append(build_refpseudovideo_dataset('train', args))
            # the ytv dataset
            args.dataset = 'ytvos'
            dataset_list.append(get_dataset("train", get_transform(args=args), args=args)[0])
            dataset = torch.utils.data.ConcatDataset(dataset_list)
        elif args.dataset == 'joint':
            dataset, _ = get_dataset('train', get_transform(args=args), args=args)
        else:
            dataset, _ = get_dataset("train", get_transform(args=args), args=args)
            if args.dataset != 'ytvos':
                dataset_test, _ = get_dataset("val", get_transform(args=args), args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    # note that if initially args.dataset is 'refs+ytvos', at this point it has been changed to 'ytvos'
    if args.dataset != 'ytvos' and args.dataset != 'joint':
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    # note that if initially args.dataset is 'refs+ytvos', at this point it has been changed to 'ytvos'
    if args.dataset != 'ytvos' and args.dataset != 'joint':
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    if args.pretrained2d_lavt_weights:
        print('Initializing video LAVT weights from pre-trained image LAVT weights.')
        model.load_from_pretrained2d_lavt_weights(args.pretrained2d_lavt_weights)
    elif args.pretrained2d_lavt_weights_for_a_3d_model:
        print('Initializing video LAVT weights from pre-trained image LAVT weights.')
        model.load_from_pretrained2d_lavt_weights_into_a_3d_model(args.pretrained2d_lavt_weights_for_a_3d_model)
    elif args.pretrained_video_lavt_weights_on_refcocos:
        print('Replacing all weights (now randomly initialized) with video LAVT weights'
              'pre-trained on the training sets (concatenated) of RefCOCO/+/g.')
        m_checkpoint = torch.load(args.pretrained_video_lavt_weights_on_refcocos, map_location='cpu')
        if args.ckpt == True:
            model.load_state_dict(m_checkpoint['model'], strict=False)
        else:
            model.load_state_dict(m_checkpoint['model'], strict=True)
    
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      find_unused_parameters=not args.use_checkpoint)
    single_model = model.module

    if not integrated:
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if not integrated:
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if not integrated:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        if args.lang_enc_params == 'encoder-10':
            params_to_optimize = [
                {'params': backbone_no_decay, 'weight_decay': 0.0},
                {'params': backbone_decay},
                {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
                # the following are the parameters of bert
                {"params": reduce(operator.concat,
                                  [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                    if p.requires_grad] for i in range(10)])},
            ]
        elif args.lang_enc_params == 'encoder-all':
            params_to_optimize = [
                {'params': backbone_no_decay, 'weight_decay': 0.0},
                {'params': backbone_decay},
                {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
                # the following are the parameters of bert
                {'params': [p for p in single_model.text_encoder.encoder.parameters() if p.requires_grad]}
            ]
            # params_to_optimize.append({'params': [p for p in single_model.text_encoder.encoder.parameters() if p.requires_grad]})
        elif args.lang_enc_params == 'embeddings+encoder-10':
            params_to_optimize = [
                {'params': backbone_no_decay, 'weight_decay': 0.0},
                {'params': backbone_decay},
                {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
                # the following are the parameters of bert
                {'params': [p for p in single_model.text_encoder.embeddings.parameters() if p.requires_grad]},
                {"params": reduce(operator.concat,
                                  [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                    if p.requires_grad] for i in range(10)])},
            ]
            '''
                params_to_optimize.append({'params': [p for p in single_model.text_encoder.embeddings.parameters() if p.requires_grad]})
                params_to_optimize.append(
                    {"params": reduce(operator.concat,
                                      [[p for p in single_model.text_encoder.encoder.layer[i].parameters() if p.requires_grad]
                                       for i in range(10)]
                                      )
                     }
                )
            '''
        elif args.lang_enc_params == 'embeddings+encoder-all':
            params_to_optimize = [
                {'params': backbone_no_decay, 'weight_decay': 0.0},
                {'params': backbone_decay},
                {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
                # the following are the parameters of bert
                {'params': [p for p in single_model.text_encoder.embeddings.parameters() if p.requires_grad]},
                {'params': [p for p in single_model.text_encoder.encoder.parameters() if p.requires_grad]}
            ]
            # params_to_optimize.append({'params': [p for p in single_model.text_encoder.embeddings.parameters() if p.requires_grad]})
            # params_to_optimize.append({'params': [p for p in single_model.text_encoder.encoder.parameters() if p.requires_grad]})
        else:
            assert False

    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    if args.fix_lr:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # loss function
    if args.loss == 'mc_dice':
        criterion = MultiClassDiceLoss()
        print("Using multi-class dice loss!")
    elif args.loss == 'dice_focal':
        criterion = DiceFocalLoss(args.loss_focal_rate, args.loss_dice_rate)
        print("Using dice & focal loss!")
    elif args.loss == 'dice_boundary':
        criterion = DiceBoundaryLoss(args.loss_boundary_rate, args.loss_dice_rate)
        print("Using dice & boundary loss!")
    else:
        criterion = cross_entropy_loss
        print("Using [0.9, 1.1] weighted cross-entropy loss!")

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1
    best_mIoU = -0.1
    # iou = 0.0  # only un-commented if we are testing model saving for YTVOS

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # optional AMP for training on the YTVOS dataset
    if args.dataset == 'ytvos' or args.image_combined_3d_pretrain or args.dataset == 'joint':
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        if args.resume:
            if "scaler" in checkpoint and args.use_amp:
                scaler.load_state_dict(checkpoint["scaler"])

    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        if args.dataset != 'ytvos' and args.dataset != 'a2d' and not args.image_combined_3d_pretrain and args.dataset != 'joint':
            train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                            iterations, bert_model)
            iou, overallIoU = evaluate(model, data_loader_test, bert_model)

            print('Average object IoU {}'.format(iou))
            print('Overall IoU {}'.format(overallIoU))

            save_checkpoint = (best_oIoU < overallIoU)
            if True:
                print('Better epoch: {}\n'.format(epoch))
                if single_bert_model is not None:
                    dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                    'lr_scheduler': lr_scheduler.state_dict()}
                else:
                    dict_to_save = {'model': single_model.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                    'lr_scheduler': lr_scheduler.state_dict()}

                # if epoch > 12:
                utils.save_on_master(dict_to_save, os.path.join('./models/', args.model_id,
                                                                'checkpoint_{:02d}_{:.2f}_{:.2f}.pth'.format(epoch, iou, overallIoU)))
                best_oIoU = overallIoU
        # elif args.dataset != 'ytvos' and not args.image_combined_3d_pretrain and args.dataset != 'joint':
        elif args.dataset == 'a2d':
            iou, overallIoU = train_one_epoch_a2d(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                            iterations, bert_model)
            iou_test, overallIoU_test, prevision_5, prevision_6, prevision_7, prevision_8, prevision_9 = evaluate_a2d(model, data_loader_test, bert_model)
            print('Test Average object IoU {}'.format(iou_test))
            print('Test Overall IoU {}'.format(overallIoU_test))
            
            print('Average object IoU {}'.format(iou))
            print('Overall IoU {}'.format(overallIoU))

            save_checkpoint = (best_oIoU < overallIoU)
            if save_checkpoint:
                print('Better epoch: {}\n'.format(epoch))
                if single_bert_model is not None:
                    dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                    'lr_scheduler': lr_scheduler.state_dict()}
                else:
                    dict_to_save = {'model': single_model.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                    'lr_scheduler': lr_scheduler.state_dict()}

                utils.save_on_master(dict_to_save, os.path.join('./models/', args.model_id,
                                                                'checkpoint_{:02d}_{:.2f}_m{:.2f}_o{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.pth'.format(epoch, iou, iou_test, overallIoU_test, prevision_5, prevision_6, prevision_7, prevision_8, prevision_9)))
                best_oIoU = iou 

        elif args.image_combined_3d_pretrain:
            _, _ = train_one_epoch_ytvos(model, criterion, optimizer, data_loader, lr_scheduler,
                                         epoch, args.print_freq, iterations, bert_model, args, scaler)
            iou, overallIoU = evaluate_ref_3d(model, data_loader_test)

            print('Average object IoU {}'.format(iou))
            print('Overall IoU {}'.format(overallIoU))

            save_checkpoint = (best_oIoU < overallIoU)
            if save_checkpoint:
                print('Better epoch: {}\n'.format(epoch))
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict(),
                                "scaler": scaler.state_dict()}
                utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                                'model_best_{}.pth'.format(args.model_id)))
                best_oIoU = overallIoU
        else:
            iou, overallIoU = train_one_epoch_ytvos(model, criterion, optimizer, data_loader, lr_scheduler,
                                                    epoch, args.print_freq, iterations, bert_model, args, scaler)
            save_checkpoint = True  # only un-commented if we are testing model saving for YTVOS
            if save_checkpoint:
                print('Saving weights... Epoch {}\n'.format(epoch))
                if single_bert_model is not None:
                    dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                    'lr_scheduler': lr_scheduler.state_dict()}
                else:
                    dict_to_save = {'model': single_model.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    "scaler": scaler.state_dict()}

                utils.save_on_master(dict_to_save, os.path.join('./models/', args.model_id,
                                                                'checkpoint_{:02d}_{:.2f}.pth'.format(epoch, iou)))
                # save last several checkpoints while training from scratch.
                if not args.pretrained_video_lavt_weights_on_refcocos and not args.dataset == 'joint':
                    utils.remove_extra_checkpoints_on_master(args) 

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
