
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_video
import json
import random
import h5py
# from pycocotools.mask import encode, area
from train import get_transform
# add
from pathlib import Path
# import datasets.transforms_video as T
from bert.tokenization_bert import BertTokenizer


def get_image_id(video_id, frame_idx, ref_instance_a2d_id):
    image_id = f'v_{video_id}_f_{frame_idx}_i_{ref_instance_a2d_id}'
    return image_id


class A2DSentencesDataset(Dataset):
    """
    A Torch dataset for A2D-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """

    def __init__(self, args, image_folder: Path, ann_file: Path, transforms, num_frames: int, subset):
        # super(A2DSentencesDataset, self).__init__() don't know why
        dataset_path = str(image_folder)
        self.mask_annotations_dir = os.path.join(dataset_path, 'Release/a2d_annotation_with_instances')
        self.videos_dir = os.path.join(dataset_path, 'Release/clips320H')
        self.ann_file = ann_file
        self.text_annotations = self.get_text_annotations()

        self._transforms = transforms
        self.num_frames = num_frames
        self.subset = subset
        self.clip_length = args.clip_length
        self.not_consecutive = args.not_consecutive

        # ADD for lavt
        self.max_tokens = 22  # 20 + 2 to account for [CLS] and [SEP]
        # bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        print(f'\n {subset} sample num: ', len(self.text_annotations))
        print('\n')

    def get_text_annotations(self):
        with open(str(self.ann_file), 'r') as f:
            text_annotations_by_frame = [tuple(a) for a in json.load(f)]
            return text_annotations_by_frame

    def __len__(self):
        return len(self.text_annotations)

    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            text_query, video_id, frame_idx, instance_id = self.text_annotations[idx]

            text_query = " ".join(text_query.lower().split())  # clean up the text query

            # read the source window frames:
            video_frames, _, _ = read_video(os.path.join(self.videos_dir, f'{video_id}.mp4'),
                                            pts_unit='sec')  # (T, H, W, C)
            vid_len = len(video_frames)
            # note that the original a2d dataset is 1 indexed, so we have to subtract 1 from frame_idx
            frame_id = frame_idx - 1

            if self.subset == 'train':
                # get a window of window_size frames with frame frame_id in the middle.
                num_frames = self.num_frames
                # random sparse sample
                sample_indx = [frame_id]
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >= global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                sample_indx.sort()
                # find the valid frame index in sampled frame list, there is only one valid frame
                valid_indices = sample_indx.index(frame_id)

            elif self.subset == 'val':
                if self.not_consecutive:
                    # get a window of window_size frames with frame frame_id in the middle.
                    # num_frames = self.clip_length
                    # random sparse sample
                    sample_indx = [frame_id]
                    # local sample
                    sample_id_before = random.randint(1, 3)
                    sample_id_after = random.randint(1, 3)
                    local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                    sample_indx.extend(local_indx)

                    # global sampling
                    if self.clip_length > 3:
                        all_inds = list(range(vid_len))
                        global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                        global_n = self.clip_length - len(sample_indx)
                        if len(global_inds) > global_n:
                            select_id = random.sample(range(len(global_inds)), global_n)
                            for s_id in select_id:
                                sample_indx.append(global_inds[s_id])
                        elif vid_len >= global_n:  # sample long range global frames
                            select_id = random.sample(range(vid_len), global_n)
                            for s_id in select_id:
                                sample_indx.append(all_inds[s_id])
                        else:
                            select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                            for s_id in select_id:
                                sample_indx.append(all_inds[s_id])
                else:
                    start_idx, end_idx = frame_id - self.clip_length // 2, frame_id + (self.clip_length + 1) // 2
                    sample_indx = []
                    for i in range(start_idx, end_idx):
                        i = min(max(i, 0), len(video_frames) - 1)  # pad out of range indices with edge frames
                        sample_indx.append(i)
                sample_indx.sort()
                # find the valid frame index in sampled frame list, there is only one valid frame
                valid_indices = sample_indx.index(frame_id)


            # read frames
            imgs, labels, boxes, masks, valid = [], [], [], [], []

            # get sentenses and attentions
            sentence_raw = text_query
            attention_mask = [0] * self.max_tokens
            padded_input_ids = [0] * self.max_tokens
            # convert words to tokens
            input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
            # possible truncation of tokens
            input_ids = input_ids[:self.max_tokens]
            padded_input_ids[:len(input_ids)] = input_ids
            attention_mask[:len(input_ids)] = [1] * len(input_ids)
            sentences_for_ref = torch.tensor(padded_input_ids).unsqueeze(0)
            attentions_for_ref = torch.tensor(attention_mask).unsqueeze(0)
            sentences = sentences_for_ref
            attentions = attentions_for_ref

            if self.subset == "train":
                n = self.num_frames
            elif self.subset == "val":
                n = self.clip_length
            else:
                assert False

            for j in range(n):
                frame_indx = sample_indx[j]
                img = F.to_pil_image(video_frames[frame_indx].permute(2, 0, 1))

                imgs.append(img)

            # read the instance mask
            frame_annot_path = os.path.join(self.mask_annotations_dir, video_id, f'{frame_idx:05d}.h5')
            f = h5py.File(frame_annot_path)
            instances = list(f['instance'])
            instance_idx = instances.index(instance_id)  # existence was already validated during init

            instance_masks = np.array(f['reMask'])
            if len(instances) == 1:
                instance_masks = instance_masks[np.newaxis, ...]
            instance_masks = torch.tensor(instance_masks).transpose(1, 2)
            f.close()

            # select the referred mask
            mask = instance_masks[instance_idx].numpy()
            if (mask > 0).any():
                valid.append(1)
            else:  # some frame didn't contain the instance
                valid.append(0)
            # mask = torch.from_numpy(mask)
            from PIL import Image
            mask = Image.fromarray(mask).convert('L')
            # labels.append(label)
            # boxes.append(box)
            masks.append(mask)

            # transform
            h, w = instance_masks.shape[-2:]

            imgs, masks = self._transforms(imgs, masks)
            imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]
            masks = torch.stack(masks, dim=0)
            target = {
                'frames_idx': torch.tensor(sample_indx),  # [T,]
                'valid_indices': torch.tensor([valid_indices]),
                'masks': masks,  # [1, H, W]
                'valid': torch.tensor(valid),  # [1,]
                'caption': text_query,
                'orig_size': torch.as_tensor([int(h), int(w)]),
                'size': torch.as_tensor([480, 480]),
                'image_id': get_image_id(video_id, frame_idx, instance_id)
            }

            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, target, sentences, attentions


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


class SimpleTransformsA2D(object):
    def __init__(self, args):
        self.per_frame_transform = get_transform(args)

    def __call__(self, images, masks):
        ret_images = []
        ret_masks = []
        for image in images:
            ret_image, ret_mask = self.per_frame_transform(image, masks[0])
            ret_images.append(ret_image)
        ret_masks.append(ret_mask)
        return ret_images, ret_masks


def build_a2d_dataset(args, image_set):
    root = Path('./data/A2D')
    assert root.exists(), f'provided A2D-Sentences path {root} does not exist'
    PATHS = {
        "train": (root, root / "Release/a2d_sentences_single_frame_train_annotations.json"),
        "val": (root, root / "Release/a2d_sentences_single_frame_test_annotations.json"),
    }
    img_folder, ann_file = PATHS[image_set]
    print(img_folder, ann_file)
    dataset = A2DSentencesDataset(args, img_folder, ann_file,
                                  transforms=SimpleTransformsA2D(args),
                                  num_frames=args.num_frames, subset=image_set)
    print(dataset)
    return dataset




