import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
from train import get_transform

from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


class ReferPseudoVideos(data.Dataset):

    def __init__(self,
                 args,
                 transforms,
                 split='train'
                 ):

        self.classes = []
        self.transforms = transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.all_samples = []
        self.max_tokens = 22
        self.num_frames = args.num_frames

        ref_ids = self.refer.getRefIds(split=self.split)
        self.ref_ids = ref_ids
        # img_ids = self.refer.getImgIds(ref_ids)
        # all_imgs = self.refer.Imgs
        # self.imgs = list(all_imgs[i] for i in img_ids)

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        for r in self.ref_ids:
            ref_id = r
            ref = self.refer.Refs[r]
            img_id = self.refer.getImgIds(r)
            img = self.refer.Imgs[img_id[0]]
            img_file_name = img['file_name']
            for el in ref['sentences']:
                # get the natural language sentence
                sentence_raw = el['raw']
                # encode words into tokens
                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                # truncate long sentences
                input_ids = input_ids[:self.max_tokens]

                # pad the sentence and construct the valid tokens mask
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens
                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1] * len(input_ids)

                # list to torch tensor
                padded_input_ids = torch.tensor(padded_input_ids).unsqueeze(0)  # (1, 22)
                attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 22)

                # construct a structure to hold information needed for each training sample pair,
                # including img file path, tokens, token masks, and "ref_ids"...
                sample = {}
                sample['exp'] = padded_input_ids
                sample['exp_att'] = attention_mask
                sample['img_name'] = img_file_name
                sample['ref_id'] = ref_id
                self.all_samples.append(sample)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        # this_ref_id = self.ref_ids[index]
        # this_img_id = self.refer.getImgIds(this_ref_id)
        # this_img = self.refer.Imgs[this_img_id[0]]
        # img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        # ref = self.refer.loadRefs(this_ref_id)
        sample = self.all_samples[index]  # 'sample' is a dictionary
        img = Image.open(os.path.join(self.refer.IMAGE_DIR, sample['img_name'])).convert("RGB")
        ref = self.refer.loadRefs(sample['ref_id'])
        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        mask = Image.fromarray(annot.astype(np.uint8), mode="P")

        imgs, masks = [], []
        ### apply transforms to construct pseudo video clip frames ###
        for _ in range(self.num_frames):
            imgs.append(img)
            masks.append(mask)

        # resize, from PIL to tensor, and mean and std normalization
        imgs, masks = self.transforms(imgs, masks)
        imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]
        masks = torch.stack(masks, dim=0)  # [T, H, W]

        target = {'masks': masks,  # [T, H, W]
                  'caption': sample['exp'],
                  'caption_mask': sample['exp_att'],
                  }

        return imgs, target


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


def build_refpseudovideo_dataset(image_set, args):
    dataset = ReferPseudoVideos(args, SimpleTransforms(args), split=image_set)
    return dataset
