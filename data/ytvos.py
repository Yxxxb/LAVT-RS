"""
Ref-YoutubeVOS dataset
"""
from pathlib import Path

import torch
# from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
# import datasets.transforms_video as T
import transforms as T
from train import get_transform

import os
from PIL import Image
import json
import numpy as np
import random

from bert.tokenization_bert import BertTokenizer

# from datasets.categories import ytvos_category_dict as category_dict


class YTVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.
    """

    def __init__(self, args, img_folder: Path, ann_file: Path, transforms, num_frames: int = 1):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        # self.return_masks = return_masks  # not used
        self.num_frames = num_frames
        # self.max_skip = max_skip
        self.max_tokens = 22  # 20 + 2 to account for [CLS] and [SEP]

        # bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        # create video meta data
        self.prepare_metas()

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        # clip number is approximate
        print('\n')

    def prepare_metas(self):
        # read object information
        # with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            # subset_metas_by_video = json.load(f)['videos']

        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            # vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                ### convert raw sentence to tokens
                sentence_raw = exp_dict['exp']
                # exp = " ".join(sentence_raw.lower().split())
                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                # token holders
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens
                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                padded_input_ids = torch.tensor(padded_input_ids).unsqueeze(0)  # (1, 22)
                attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 22)

                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = padded_input_ids
                    meta['exp_att'] = attention_mask
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # # get object category
                    # obj_id = exp_dict['obj_id']
                    # meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)

    '''
    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2
    '''

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            #video, exp, obj_id, category, frames, frame_id = \
            #    meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
            video, exp, exp_att, obj_id, frames, frame_id = \
                meta['video'], meta['exp'], meta['exp_att'], meta['obj_id'], meta['frames'], meta['frame_id']

            # clean up the caption
            # exp = " ".join(exp.lower().split())
            # category_id = category_dict[category]
            vid_len = len(frames)

            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            if self.num_frames != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                # Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)  # unsorted
                # at this point we have exactly 3 sampled frames centering around sample_indx;
                # note that they may include two 0th frames or two (vid_len-1)th frames
                # the interval [1, 3] is actually a hyper-parameter

                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    # global_inds can't be empty, although either half can be empty
                    global_n = num_frames - len(sample_indx)  # how many extra frames we need to sample
                    # we have enough frames in the pool for selection
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        # random.sample(population, k, *, counts=None)
                        # Return a k-length list of unique elements chosen from the population sequence or set.
                        # Used for random sampling without replacement. If the population contains repeats,
                        # then each occurrence is a possible selection in the sample.
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])  # still unsorted; can actually use .extend()
                    # we do not have enough (or just enough) but the whole video has enough,
                    # then allow repeats in sampled frames
                    elif vid_len >= global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])  # don't actually need to index, index is the same as
                            # the indexed elements. but it is good reminder that in our previous case this is not so.
                            # can also use .extend()
                    # even the whole video sequence does not have enough frames left
                    # then use the whole video sequence and some. This case has the largest number of repeats naturally
                    else:
                        multiple = global_n // vid_len
                        select_id = random.sample(range(vid_len), global_n % vid_len) + (list(range(vid_len)) * multiple)
#                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])  # indexing also not necessary here; can use .extend()
            sample_indx.sort()  # .sort() is in-place and sorted(x) is not in-place
            # note that len(sample_indx) == self.num_frames is true here

            # read frames and masks
            # imgs, labels, boxes, masks, valid = [], [], [], [], []
            imgs, masks, valid = [], [], []
            for j in range(self.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('P')
                
                 
                # create the target
                # label = torch.tensor(category_id)
                mask = np.array(mask)
                mask = (mask == obj_id).astype(np.float32)  # 0,1 binary
                if (mask > 0).any():
                    # y1, y2, x1, x2 = self.bounding_box(mask)
                    # box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else:  # some frames don't contain the instance
                    # box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)

                # mask = torch.from_numpy(mask)
                mask = Image.fromarray(mask).convert('L')

                # append
                imgs.append(img)
                # labels.append(label)
                masks.append(mask)
                # boxes.append(box)

            # transform
            w, h = img.size  # Pillow image size is (width, height); the original image resolution
            # assuming all frames in the same vid have the same shape
            # labels = torch.stack(labels, dim=0)
            # boxes = torch.stack(boxes, dim=0)
            # stack by default prepends a new dim and then concatenates the given list along the new dim
            # boxes[:, 0::2].clamp_(min=0, max=w)
            # boxes[:, 1::2].clamp_(min=0, max=h)
            # masks = torch.stack(masks, dim=0)

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            # imgs, target = self._transforms(imgs, target)
            # deliberately sending *lists* into the transform-function chain, which also returns lists
            imgs, masks = self._transforms(imgs, masks)
            imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W], if single images, then T=1
            masks = torch.stack(masks, dim=0)  # [T, H, W], if single images, then T=1

            target = {
                'frames_idx': torch.tensor(sample_indx),  # [T,]
                # 'labels': labels,  # [T,]
                # 'boxes': boxes,  # [T, 4], xyxy
                'masks': masks,  # [T, H, W]
                'valid': torch.tensor(valid),  # [T,]
                'caption': exp,
                'caption_mask': exp_att,
                'orig_size': torch.as_tensor([int(h), int(w)])
            }

            # have postponed validity check till after clip sampling has been completed once;
            # the logic and code are simpler this way but this is less efficient, just by how much depends on the
            # average frequency of out-of-view (disappearance) occurrences
            if torch.any(target['valid'] == 1):  # at least one instance; then break out of the loop
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)
                # we generate a random idx internally and repeat the frames sampling process again

        return imgs, target


### Video wrapper ###
class SimpleTransforms(object):
    def __init__(self, args):
        self.per_frame_transform = get_transform(args)

    def __call__(self, images, masks):
        ret_images = []
        ret_masks = []
        for image, mask in zip(images, masks):
            ret_image, ret_mask = self.per_frame_transform(image, mask)
            ret_images.append(ret_image)
            ret_masks.append(ret_mask)

        return ret_images, ret_masks


'''
def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])

    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
'''

'''
def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),  # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size),
                           num_frames=args.num_frames, max_skip=args.max_skip)
    return dataset
'''


def build_ytvos_dataset(args):
    root = Path(args.ytvos_data_root)
    assert root.exists(), f'the provided YT-VOS path {root} does not exist'
    img_folder, ann_file = (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json")
    dataset = YTVOSDataset(args, img_folder, ann_file, transforms=SimpleTransforms(args), num_frames=args.num_frames)
    return dataset
