'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import json
import random
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F

import multiprocessing as mp

import threading

from colormap import colormap
from lib import segmentation
import utils

from bert.tokenization_bert import BertTokenizer

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


# build transform
def get_transform(args):
    return T.Compose([T.Resize((args.img_size, args.img_size)),
                      T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                     )


#transform = T.Compose([T.Resize((480, 480)),
#                      T.ToTensor(),
#                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
#                     )

lock = threading.Lock()


def main(args):
    print("Inference only supports batch size 1")

    # fix the seed for reproducibility
    seed = 0 + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split  # 'valid' or 'test' (mostly should be 'valid')
    # save path
    if args.pretrained2d_lavt_weights:
        # './checkpoints/model_best_....pth'
        model_name = args.pretrained2d_lavt_weights.split('/')[-1]  # the last component, the weight file name
        model_name = model_name[11:-4]  # remove 'model_best_' and '.pth'
        output_dir = './ytv_results/' + model_name  # we chose to save only the best epoch for image LAVT models
    else:
        assert args.resume
        resume_path_components = args.resume.split('/')
        model_name = resume_path_components[-2]  # lavt_...
        checkpoint_name = resume_path_components[-1].rpartition('.')[0]  # checkpoint name; split once from right by '.'
        output_dir = './ytv_results/' + model_name + '/' + checkpoint_name
    save_path_prefix = os.path.join(output_dir, split)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    root = Path(args.ytvos_data_root)  # './data/ReferringYouTubeVOS2021/'
    img_folder = os.path.join(root, split, "JPEGImages")  # split - valid
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    # For some reason the competition's validation expressions dict contains both the validation (202) &
    # test videos (305). So we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos  # set subtraction
    video_list = sorted([video for video in valid_videos])  # sort video names in ascending order
    assert len(video_list) == 202, 'error: incorrect number of validation videos'

    # get image transformations
    transform = get_transform(args)

    # create subprocess
    thread_num = args.ngpus
    #global result_dict
    #result_dict = mp.Manager().dict()

    processes = []
    #lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        # choose the evaluation method,
        # according to whether the model is image-version lavt_one or video-version lavt_video,
        if args.model == 'lavt_video' or args.ytvos_2d_swin_3d_pwam:
            p = mp.Process(target=evaluate_whole_videos, args=(i, args, data,
                                                               save_path_prefix, save_visualize_path_prefix,
                                                               img_folder, sub_video_list, transform
                                                               )
                           )
        else:
            p = mp.Process(target=evaluate_single_frames, args=(i, args, data,
                                                                save_path_prefix, save_visualize_path_prefix,
                                                                img_folder, sub_video_list, transform
                                                                )
                           )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    #result_dict = dict(result_dict)
    #num_all_frames_gpus = 0
    #for pid, num_all_frames in result_dict.items():
    #    num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" % total_time)


def evaluate_whole_videos(pid, args, data, save_path_prefix,
                          save_visualize_path_prefix, img_folder, video_list, transform):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    # model
    if pid == 0:
        print(args.model)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    max_tokens = 22  # 20 + 2 to account for [CLS] and [SEP]

    single_model = segmentation.__dict__[args.model](pretrained='', args=args)

    if args.pretrained2d_lavt_weights:
        single_model.load_from_pretrained2d_lavt_weights(args.pretrained2d_lavt_weights)
    else:
        assert args.resume
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'], strict=False)
        # strict=False is only to accommodate the case that the trained weights are from
        # a model trained using gradient checkpointing (e.g., video base). In this case, as we do not specify
        # --use_checkpoint during evaluation, the initialized model would have an extra LG module in the last layer
        # which is not used in anyway, but setting strict=True in this case would throw an error...
    model = single_model.to(torch.device('cuda'))  # current cuda device
    n_parameters = sum(p.numel() for p in single_model.parameters() if p.requires_grad)
    if pid == 0:
        print('number of params:', n_parameters)

    model.eval()

    # 1. For each video
    for video in video_list:
        metas = []  # list[dict], length is number of expressions
        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        # video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            sentence_raw = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            # encode raw sentence and get the sentence valid positions mask
            input_ids = tokenizer.encode(text=sentence_raw, add_special_tokens=True)
            # token holders
            attention_mask = [0] * max_tokens
            padded_input_ids = [0] * max_tokens
            # truncation of tokens
            input_ids = input_ids[:max_tokens]
            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1] * len(input_ids)

            sentence = torch.tensor(padded_input_ids).unsqueeze(0)  # creates the batch dimension. (1, 22)
            attention = torch.tensor(attention_mask).unsqueeze(0)  # creates the batch dimension. (1, 22)
            sentence = sentence.to(torch.device('cuda'))
            attention = attention.to(torch.device('cuda'))

            video_len = len(frames)
            # store images
            imgs = []
            # open and transform each image, and then stack them to form the clip input to our model
            for t in range(video_len):
                frame = frames[t]
                img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                img = Image.open(img_path).convert('RGB')
                origin_w, origin_h = img.size  # assuming all frames in the same video have the same shape
                imgs.append(transform(img))  # list of (3, 480, 480)

            imgs = torch.stack(imgs, dim=0).to(torch.device('cuda'))  # [video_len, 3, h, w]
            if not args.ytvos_2d_swin_3d_pwam:
                imgs = imgs.unsqueeze(0)  # [1, T, 3, H, W] per our model's requirement
            # whole video inference
            with torch.no_grad():
                outputs = model(imgs, sentence, l_mask=attention)  # (1*T, 2, 480, 480)
                outputs = F.interpolate(outputs, size=(origin_h, origin_w), mode='bilinear', align_corners=True)
                # (1*T, 2, origin_h, origin_w)
                # Note: should try whether align corners or not; ReferFormer set to False; I stick with the more
                # traditional True; but unsure about this.
            pred_masks = outputs.argmax(1).cpu().data.numpy()  # (1*T, orig_h, orig_w)

            if args.visualize:
                for t, frame in enumerate(frames):
                    # get the original image
                    img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                    source_img = Image.open(img_path).convert('RGBA')  # PIL image
                    # draw mask
                    source_img = vis_add_mask(source_img, pred_masks[t], color_list[i % len(color_list)])

                    # save
                    save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                    if not os.path.exists(save_visualize_path_dir):
                        os.makedirs(save_visualize_path_dir)
                    save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                    source_img.save(save_visualize_path)

            # save binary image
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(video_len):
                frame_name = frames[j]  # frame_name is the same as the 'frame' that we have seen before
                mask = pred_masks[j].astype(np.float32)  # (orig_h, orig_w)
                mask = Image.fromarray(mask * 255).convert('L')  # to greyscale image. *255 shouldn't be necessary
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)

        with lock:
            progress.update(1)

    with lock:
        progress.close()


def evaluate_single_frames(pid, args, data, save_path_prefix,
                           save_visualize_path_prefix, img_folder, video_list, transform):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    # model
    if pid == 0:
        print(args.model)

    '''
    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)
    '''

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    max_tokens = 22  # 20 + 2 to account for [CLS] and [SEP]

    single_model = segmentation.__dict__[args.model](pretrained='', args=args)

    # if not (args.ytvos_2d_swin_pwam or args.ytvos_2d_swin_3d_pwam):
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
        
    model = single_model.to(torch.device('cuda'))  # current cuda device
    # model_without_ddp = model
    n_parameters = sum(p.numel() for p in single_model.parameters() if p.requires_grad)
    if pid == 0:
        print('number of params:', n_parameters)

    #if args.resume:
    #    checkpoint = torch.load(args.resume, map_location='cpu')
    #    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    #    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    #    if len(missing_keys) > 0:
    #        print('Missing Keys: {}'.format(missing_keys))
    #    if len(unexpected_keys) > 0:
    #        print('Unexpected Keys: {}'.format(unexpected_keys))
    #else:
    #    raise ValueError('Please specify the checkpoint for inference.')

    # start inference
    #num_all_frames = 0
    model.eval()

    # 1. For each video
    for video in video_list:
        metas = []  # list[dict], length is number of expressions

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        # video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            sentence_raw = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            # encode raw sentence and get the sentence valid positions mask
            input_ids = tokenizer.encode(text=sentence_raw, add_special_tokens=True)
            # token holders
            attention_mask = [0] * max_tokens
            padded_input_ids = [0] * max_tokens
            # truncation of tokens
            input_ids = input_ids[:max_tokens]
            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1] * len(input_ids)

            sentence = torch.tensor(padded_input_ids).unsqueeze(0)  # creates the batch dimension. (1, 22)
            attention = torch.tensor(attention_mask).unsqueeze(0)  # creates the batch dimension. (1, 22)
            sentence = sentence.to(torch.device('cuda'))
            attention = attention.to(torch.device('cuda'))

            video_len = len(frames)
            # store images
            # imgs = []
            # 3. For each frame
            for t in range(video_len):
                frame = frames[t]
                img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                img = Image.open(img_path).convert('RGB')
                origin_w, origin_h = img.size
                #imgs.append(get_transform(args)(img))  # list[img]
                img = transform(img)  # (3, 480, 480)
                img = img.unsqueeze(0)  # (1, 3, 480, 480)
                img = img.to(torch.device('cuda'))

                # per-frame model inference
                # imgs = torch.stack(imgs, dim=0).to(torch.device('cuda'))  # [video_len, 3, h, w]
                #img_h, img_w = imgs.shape[-2:]
                #size = torch.as_tensor([int(img_h), int(img_w)]).to(torch.device('cuda'))
                #target = {"size": size}

                with torch.no_grad():
                    # outputs = model([imgs], [exp], [target])
                    output = model(img, sentence, l_mask=attention)  # (1, 2, 480, 480)
                    output = F.interpolate(output, size=(origin_h, origin_w), mode='bilinear', align_corners=True)
                    ## Note: should try whether align corners or not; ReferFormer set to False; I stick with the more
                    ## traditional True; but unsure.
                    pred_mask = output.argmax(1).cpu().data.numpy()  # (1, orig_h, orig_w)

                #pred_logits = outputs["pred_logits"][0]
                # pred_boxes = outputs["pred_boxes"][0]
                #pred_masks = outputs["pred_masks"][0]
                # pred_ref_points = outputs["reference_points"][0]

                # according to pred_logits, select the query index
                # pred_scores = pred_logits.sigmoid()  # [t, q, k]
                # pred_scores = pred_scores.mean(0)  # [q, k]
                # max_scores, _ = pred_scores.max(-1)  # [q,]
                # _, max_ind = max_scores.max(-1)  # [1,]
                # max_inds = max_ind.repeat(video_len)
                # pred_masks = pred_masks[range(video_len), max_inds, ...]  # [t, h, w]
                # pred_masks = pred_masks.unsqueeze(0)

                # pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()  # [t, h, w]

                # store the video results
                # all_pred_logits = pred_logits[range(video_len), max_inds]
                # all_pred_boxes = pred_boxes[range(video_len), max_inds]
                # all_pred_ref_points = pred_ref_points[range(video_len), max_inds]
                # all_pred_masks = pred_masks

                if args.visualize:
                    # for t, frame in enumerate(frames):
                    # original
                    img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                    source_img = Image.open(img_path).convert('RGBA')  # PIL image

                    #draw = ImageDraw.Draw(source_img)
                    #draw_boxes = all_pred_boxes[t].unsqueeze(0)
                    #draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()

                    # draw boxes
                    #xmin, ymin, xmax, ymax = draw_boxes[0]
                    #draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[i % len(color_list)]),
                    #               width=2)

                    # draw reference point
                    #ref_points = all_pred_ref_points[t].unsqueeze(0).detach().cpu().tolist()
                    #draw_reference_points(draw, ref_points, source_img.size, color=color_list[i % len(color_list)])

                    # draw mask
                    # source_img = vis_add_mask(source_img, all_pred_masks[t], color_list[i % len(color_list)])
                    source_img = vis_add_mask(source_img, pred_mask[0], color_list[i % len(color_list)])

                    # save
                    save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                    if not os.path.exists(save_visualize_path_dir):
                        os.makedirs(save_visualize_path_dir)
                    save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                    source_img.save(save_visualize_path)

                # save binary image
                save_path = os.path.join(save_path_prefix, video_name, exp_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # for j in range(video_len):
                # frame_name = frames[j]
                # frame_name is the same as frame
                # mask = all_pred_masks[j].astype(np.float32)
                mask = pred_mask[0].astype(np.float32)  # (orig_h, orig_w)
                mask = Image.fromarray(mask * 255).convert('L')  # to greyscale image. *255 shouldn't be necessary
                # save_file = os.path.join(save_path, frame_name + ".png")
                save_file = os.path.join(save_path, frame + ".png")
                mask.save(save_file)

        with lock:
            progress.update(1)

    # result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


# visuaize functions
'''
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x - 10, y, x + 10, y), tuple(cur_color), width=4)
        draw.line((x, y - 10, x, y + 10), tuple(cur_color), width=4)


def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=tuple(cur_color), outline=tuple(cur_color), width=1)
'''


def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')  # np. reshape should not be necessary here
    mask = mask > 0.5  # 0 ahould also do; any non-zero value should do

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img


if __name__ == "__main__":
    mp.set_start_method('spawn')
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
