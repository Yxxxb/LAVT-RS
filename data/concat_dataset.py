# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.utils.data

from torch.utils.data import Dataset, ConcatDataset
from .refexp2seq import build as build_seq_refexp
from .ytvos import build_ytvos_dataset



def build_joint_dataset(image_set, args):
    concat_data = []

    print('preparing coco2seq dataset ....')
    coco_names = ["refcoco", "refcoco+", "refcocog"]
    for name in coco_names:
        coco_seq =  build_seq_refexp(name, image_set, args)
        concat_data.append(coco_seq)

    print('preparing ytvos dataset  .... ')
    ytvos_dataset = build_ytvos_dataset(args)
    concat_data.append(ytvos_dataset)

    concat_data = ConcatDataset(concat_data)
    print('finish preparing joint train dataset!')

    return concat_data
