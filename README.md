# LAVT: Language-Aware Vision Transformer for Referring Segmentation

[Zhao Yang](https://github.com/yz93)\*, [Jiaqi Wang](https://myownskyw7.github.io/)\*, [Xubing Ye](https://github.com/Yxxxb)\*, [Yansong Tang](https://andytang15.github.io), [Kai Chen](https://chenkai.site/), [Hengshuang Zhao](https://hszhao.github.io/), [Philip H.S. Torr](https://scholar.google.com/citations?user=kPxa2w0AAAAJ&hl=zh-CN)

*\* Equal Contribution*.

Welcome to the repository for the method presented in
"Language-Aware Vision Transformer for Referring Segmentation."


Code in this repository is written using [PyTorch](https://pytorch.org/) and is organized in the following way (assuming the working directory is the root directory of this repository):

* `./lib` contains files implementing the main network.
* `./bert` contains files migrated from [Hugging Face Transformers v3.0.2](https://huggingface.co/transformers/v3.0.2/quicktour.html),
  which implement the BERT language model.
  We have used Transformers v3.0.2 during development but it has a bug that would appear when using `DistributedDataParallel`.
  Therefore we decided to maintain a copy of the relevant source files in this repository.
  This way, the bug is fixed and code in this repository is self-contained.
* `./refer` contains data pre-processing code and is also where data should be placed, including the images and all annotations.
  It is cloned from [refer](https://github.com/lichengunc/refer). 
* `./data/dataset_refer_bert.py` is where the dataset class is defined.
* `./utils.py` defines functions that track statistics during training and also setup
  functions for using `DistributedDataParallel`.
* Inside `./lib`, `_utils.py` defines the highest-level model, which incorporates the backbone network
  defined in `backbone.py` and the simple mask decoder defined in `mask_predictor.py`,
  and `segmentation.py` provides a model interface and functions used to initialize the model.
* `./lib/video_swin_transformer.py` contains the new Video Swin visual backbone.


--------------------------------------------

* `./train.py` is invoked to train the model.
* `./test_ytvos.py` is invoked to run inference on
  the validation set of YouTube-VOS. The output prediction masks folder
  needs to be renamed and compressed to .zip and uploaded to the
  2022 competition server for evaluation.


## Setting Up

### Preliminaries

The code has been verified to work with PyTorch v1.7.1/v1.8.1 and Python 3.7.

1. Clone this repository.
2. Change directory to root of this repository.

### Package Dependencies

1. Create a new Conda environment with Python 3.7 then activate it:

```shell
conda create -n lavt python==3.7
conda activate lavt
```

2. Install PyTorch v1.7.1 with a CUDA version that works on your cluster/machine (CUDA 10.2 is used in this example):

```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the packages in `requirements.txt` via `pip`:

```shell
pip install -r requirements.txt
```

### Datasets

----------------------------------

#### Image

----------------------------------

1. Follow instructions in the `./refer` directory to set up subdirectories
   and download annotations.
   This directory is a git clone (minus two data files that we do not need)
   from the [refer](https://github.com/lichengunc/refer) public API.

2. Download images from [COCO](https://cocodataset.org/#download).
   Please use the first downloading link *2014 Train images [83K/13GB]*, and extract
   the downloaded `train_2014.zip` file to `./refer/data/images/mscoco/images`.

#### Video

Data directories have the following structure:

```text
lavt_video/
└── data/
    ├── A2D/
    │   └── Release/
    │       ├── a2d_annotation.txt
    │       ├── a2d_missed_videos.txt
    │       ├── videoset.csv
    │       ├── a2d_annotation_with_instances/  # ls -l | wc -l gives 3756
    │       │   └── */  (video folders)
    │       │       └── *.h5  (mask annotation files) 
    │       ├── Annotations/
    │       │   ├── col  # ls -l | wc -l gives 3783
    │       │   │   └── */ (video folders)
    │       │   │       └── *.png  (masks in png format) 
    │       │   └── mat  # ls -l | wc -l gives 3783
    │       │       └── */ (video folders)
    │       │           └── *.mat  (masks stored as matrices)
    │       ├── pngs320H/  # ls -l | wc -l gives 3783
    │       │   └── */ (video folders)
    │       │       └── *.png  (frame images; index starts at 00001)
    │       └── clips320H/  # ls -l | wc -l gives 3783
    │           └── *.mp4  (raw MP4 videos)
    │
    │
    └── ReferringYouTubeVOS2021/
        ├── train/
        │    ├── Annotations/  # ls -l | wc -l gives 3472
        │    │   └── */  (video folders)
        │    │       └── *.png  (mask images)
        │    ├── JPEGImages/  # ls -l | wc -l gives 3472
        │    │   └── */  (video folders)
        │    │       └── *.jpg  (frame images)
        │    └── meta.json  # (this is 2019 training set meta file; has no expressions)
        │
        ├── valid/
        │   └── JPEGImages/  # ls -l | wc -l gives 203
        │       └── */  (video folders)
        │           └── *.jpg  (frame images)
        ├── test/ 
        │   └── JPEGImages/  # ls -l | wc -l gives 306
        │       └── */  (video folders)
        │           └── *.jpg  (frame images)
        └── meta_expressions/
            ├── train/
            │   └── meta_expressions.json  (video meta info with expressions)
            ├── valid/
            │   └── meta_expressions.json  (video meta info with expressions)
            └── test/
                └── meta_expressions.json  (video meta info with expressions)
```

### Weights for Training

1.  Create the `./pretrained_weights` directory where we will be storing the weights.

```shell
mkdir ./pretrained_weights
```

2.  1. The original Swin Transformer. Download [pre-trained classification weights of
       the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth), `swin_base_patch4_window12_384_22k.pth`, into `./pretrained_weights`.
       These weights are needed in training to initialize the model.
    2. The Video Swin Transformer. Download `swin_tiny_patch244_window877_kinetics400_1k.pth`,
       `swin_small_patch244_window877_kinetics400_1k.pth`, `swin_base_patch244_window877_kinetics400_1k.pth`,
       `swin_base_patch244_window877_kinetics400_22k.pth`, `swin_base_patch244_window877_kinetics600_22k.pth`,
       and `swin_base_patch244_window1677_sthv2.pth` into `./pretrained_weights`.


3. Create the `./checkpoints` directory where the program will save the weights during training.
   (this is only true for the image-version LAVT; video LAVT saves 10 currently best checkpoints in `./models/[args.model_id]`).

```shell
mkdir ./checkpoints
```

## Training

We use `DistributedDataParallel` from PyTorch.
The released `lavt` weights were trained using 4 x 32G V100 cards (max mem on each card was about 26G).
The released `lavt_one` weights were trained using 8 x 32G V100 cards (max mem on each card was about 13G).
The released `lavt_video` weights were trained using 8 x 32G V100 cards (max mem on each card was about 13G).
Using more cards was to accelerate training.

To run on 4 GPUs (with IDs 0, 1, 2, and 3) on a single node for RIS:

```shell
mkdir ./models

mkdir ./models/refcoco
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcoco --model_id refcoco --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco/output

mkdir ./models/refcoco+
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcoco+ --model_id refcoco+ --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco+/output

mkdir ./models/gref_umd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcocog --splitBy umd --model_id gref_umd --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/gref_umd/output

mkdir ./models/gref_google
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcocog --splitBy google --model_id gref_google --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/gref_google/output
```

To run on 8 GPUs (with IDs 0, 1, 2, 3, 4, 5, 6, 7) on a single node for RVOS:

```shell
mkdir ./models

mkdir ./models/a2d
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 train.py --model lavt_video --dataset a2d --model_id a2d --batch-size 4 --lr 0.00006 --wd 1e-2 --swin_type tiny --sep_t_pwam --conv3d_kernel_size_t 3-3-3 --conv3d_kernel_size_s 1-1-1 --w_t3x3_s1x1 --mm_t3x3_s1x1 --pretrained_swin_weights ./pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/a2d/output

mkdir ./models/ytvos
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 train.py --model lavt_video --dataset a2d --model_id a2d --batch-size 1 --lr 0.00005 --wd 1e-2 --swin_type tiny --sep_t_pwam --conv3d_kernel_size_t 3-3-3 --conv3d_kernel_size_s 1-1-1 --w_t3x3_s1x1 --mm_t3x3_s1x1 --pretrained_swin_weights ./pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth --epochs 30 --img_size 480 2>&1 | tee ./models/a2d/output
```

* *--model* is a pre-defined model name. Options include `lavt` , `lavt_video`and `lavt_one`. See [Updates](#updates).
* *--dataset* is the dataset name. One can choose from `refcoco`,`refcoco+`, and `refcocog`.
* *--splitBy* needs to be specified if and only if the dataset is G-Ref (which is also called RefCOCOg).
  `umd` identifies the UMD partition and `google` identifies the Google partition.
* *--model_id* is the model name one should define oneself (*e.g.*, customize it to contain training/model configurations, dataset information, experiment IDs, *etc*.).
  It is used in two ways: Training log will be saved as `./models/[args.model_id]/output` and the best checkpoint will be saved as `./checkpoints/model_best_[args.model_id].pth`.
* *--swin_type* specifies the version of the Swin Transformer.
  One can choose from `tiny`, `small`, `base`, and `large`. The default is `base`.
* *--pretrained_swin_weights* specifies the path to pre-trained Swin Transformer weights used for model initialization.
* Note that currently we need to manually create the `./models/[args.model_id]` directory via `mkdir` before running `train.py`.
  This is because we use `tee` to redirect `stdout` and `stderr` to `./models/[args.model_id]/output` for logging.
  This is a nuisance and should be resolved in the future, *i.e.*, using a proper logger or a bash script for initiating training.


## Testing

For RefCOCO/RefCOCO+, run one of

```shell
python test.py --model lavt --swin_type base --dataset refcoco --split val --resume ./checkpoints/refcoco.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
python test.py --model lavt --swin_type base --dataset refcoco+ --split val --resume ./checkpoints/refcoco+.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
```

* *--split* is the subset to evaluate, and one can choose from `val`, `testA`, and `testB`.
* *--resume* is the path to the weights of a trained model.

For G-Ref (UMD)/G-Ref (Google), run one of

```shell
python test.py --model lavt --swin_type base --dataset refcocog --splitBy umd --split val --resume ./checkpoints/gref_umd.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
python test.py --model lavt --swin_type base --dataset refcocog --splitBy google --split val --resume ./checkpoints/gref_google.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
```

* *--splitBy* specifies the partition to evaluate.
  One can choose from `umd` or `google`.
* *--split* is the subset (according to the specified partition) to evaluate, and one can choose from `val` and `test` for the UMD partition, and only `val` for the Google partition..
* *--resume* is the path to the weights of a trained model.

For A2D, run

```shell
python test.py --model lavt_video --swin_type tiny --dataset a2d --conv3d_kernel_size_t 3-3-3 --conv3d_kernel_size_s 1-1-1 --w_t3x3_s1x1 --mm_t3x3_s1x1 --num_frames 8 --split val --resume ./checkpoints/a2d.pth --sample_3 --img_size 480 --clip_length 16 --split val
```

* *--split* is the subset to evaluate, and one can only choose `val`.
* *--resume* is the path to the weights of a trained model.
* --*clip_length* is the length of frames of each clip while testing.

For YTVOS, run

```shell
python test_ytvos.py 1 --model lavt_video --sep_t_pwam --conv3d_kernel_size_t 3-3-3 --conv3d_kernel_size_s 1-1-1 --w_t3x3_s1x1 --mm_t3x3_s1x1 --swin_type tiny --dataset ytvos --split valid --resume ./models/ytvos.pth --img_size 480
```

* *--split* is the subset to evaluate, and one can only choose `val`.
* *--resume* is the path to the weights of a trained model.

## Results and weights

#### Image

----------------------------------

The complete test results of the released LAVT models are summarized as follows: we report the results of LAVT trained with a multi-class Dice loss and based on the new implementation (`lavt_one`).

|      Dataset      | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | Overall IoU | Mean IoU |
| :---------------: | :---: | :---: | :---: | :---: | :---: | :---------: | :------: |
|    RefCOCO val    | 85.87 | 82.13 | 76.64 | 65.45 | 35.30 |    73.50    |  75.41   |
|  RefCOCO test A   | 88.47 | 85.63 | 80.57 | 68.84 | 35.71 |    75.97    |  77.31   |
|  RefCOCO test B   | 80.20 | 76.49 | 70.34 | 60.12 | 34.94 |    69.33    |  71.86   |
|   RefCOCO+ val    | 76.19 | 72.27 | 66.82 | 56.87 | 30.15 |    63.79    |  67.65   |
|  RefCOCO+ test A  | 82.50 | 79.44 | 74.00 | 63.27 | 31.99 |    69.79    |  72.53   |
|  RefCOCO+ test B  | 68.03 | 63.35 | 57.29 | 47.92 | 26.98 |    56.49    |  61.22   |
|  G-Ref val (UMD)  | 75.82 | 71.06 | 63.99 | 52.98 | 27.31 |    64.02    |  67.41   |
| G-Ref test (UMD)  | 76.12 | 71.13 | 64.58 | 53.62 | 28.03 |    64.49    |  67.45   |
| G-Ref val (Goog.) | 72.57 | 68.65 | 63.09 | 53.33 | 28.14 |    61.31    |  64.84   |

To train weights of image LAVT for testing, you could follow:

1. Create the `./checkpoints` directory where we will be storing the weights.

```shell
mkdir ./checkpoints
```

2. Download LAVT model weights (which are stored on Google Drive) using links below and put them in `./checkpoints`.

| [RefCOCO](https://drive.google.com/file/d/13D-OeEOijV8KTC3BkFP-gOJymc6DLwVT/view?usp=sharing) | [RefCOCO+](https://drive.google.com/file/d/1B8Q44ZWsc8Pva2xD_M-KFh7-LgzeH2-2/view?usp=sharing) | [G-Ref (UMD)](https://drive.google.com/file/d/1BjUnPVpALurkGl7RXXvQiAHhA-gQYKvK/view?usp=sharing) | [G-Ref (Google)](https://drive.google.com/file/d/1weiw5UjbPfo3tCBPfB8tu6xFXCUG16yS/view?usp=sharing) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

3. Model weights and training logs of the new lavt_one implementation are below.

|                           RefCOCO                            |                           RefCOCO+                           |                         G-Ref (UMD)                          |                        G-Ref (Google)                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [log](https://drive.google.com/file/d/1YIojIHqe3bxxsWOltifa2U9jH67hPHLM/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1xFMEXr6AGU97Ypj1yr8oo00uObbeIQvJ/view?usp=sharing) | [log](https://drive.google.com/file/d/1Z34T4gEnWlvcSUQya7txOuM0zdLK7MRT/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1HS8ZnGaiPJr-OmoUn4-4LVnVtD_zHY6w/view?usp=sharing) | [log](https://drive.google.com/file/d/14VAgahngOV8NA6noLZCqDoqaUrlW14v8/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/14g8NzgZn6HzC6tP_bsQuWmh5LnOcovsE/view?usp=sharing) | [log](https://drive.google.com/file/d/1JBXfmlwemWSvs92Rky0TlHcVuuLpt4Da/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1IJeahFVLgKxu_BVmWacZs3oUzgTCeWcz/view?usp=sharing) |

* The Prec@K, overall IoU and mean IoU numbers in the training logs will differ from the final results obtained by running `test.py`, because only one out of multiple annotated expressions is randomly selected and evaluated for each object during training. But these numbers give a good idea about the test performance. The two should be fairly close.

#### Video

------

Results on the Refer-YouTube-VOS dataset under the “train-from-scratch” training setting with different backbone networks employed.

| Backbone     | J & F | J     | F     |
| ------------ | ----- | ----- | ----- |
| Video Swin-T | 57.04 | 55.39 | 58.69 |
| Video Swin-S | 58.79 | 57.10 | 60.49 |
| Video Swin-B | 60.45 | 58.49 | 62.42 |

Results on the Refer-YouTube-VOS dataset under the “pretrain-then-finetune” training setting with different backbone networks employed.

| Backbone     | J & F | J     | F     |
| ------------ | ----- | ----- | ----- |
| Video Swin-T | 60.91 | 59.37 | 62.45 |
| Video Swin-S | 62.96 | 60.35 | 65.56 |
| Video Swin-B | 64.90 | 62.22 | 67.58 |

Results on the A2D-Sentences dataset under the “train-from-scratch” training setting with different backbone networks employed.

| Backbone     | oIoU | mIoU |
| ------------ | ---- | ---- |
| Video Swin-T | 74.4 | 65.9 |
| Video Swin-S | 75.5 | 67.7 |
| Video Swin-B | 77.0 | 68.7 |

Results on the A2D-Sentences dataset under the “pretrain-then-finetune” training setting with different backbone networks employed.

| Backbone     | oIoU | mIoU |
| ------------ | ---- | ---- |
| Video Swin-T | 77.9 | 70.0 |
| Video Swin-S | 79.1 | 70.4 |
| Video Swin-B | 80.7 | 71.9 |

You could download video LAVT model weights (which are stored on Tsinghua cloud disk) using links below and put them in `./checkpoints`.

| Refer-YouTube-VOS | A2D-Sentences |
| --------------------- | ----------------- |
| [Refcoco_pretrain](https://cloud.tsinghua.edu.cn/d/54044571899a4d0ebffd/) | [Refcoco_pretrain](https://cloud.tsinghua.edu.cn/d/54044571899a4d0ebffd/) |
| [YTVOS_finetune](https://cloud.tsinghua.edu.cn/d/54d4499c22af48559fce/) | [A2D_finetune](https://cloud.tsinghua.edu.cn/d/282406a0385b4a62b857/) |
| [YTVOS_scratch]() | [A2D_scratch](https://cloud.tsinghua.edu.cn/d/51dd4e5c1dea41428b45/) |
| [3D_PWAM_ablation](https://cloud.tsinghua.edu.cn/d/7d93ecb3e5e3423882ac/) | - |
| [CM-FPN_ablation](https://cloud.tsinghua.edu.cn/d/1c00295c054d4b578e16/) | - |

## Contributing

We appreciate all contributions. It helps the project if you could

- report issues you are facing,
- give a :+1: on issues reported by others that are relevant to you,
- answer issues reported by others for which you have found solutions,
- and implement helpful new features or improve the code otherwise with pull requests.

## Acknowledgements

Code in this repository is built upon several public repositories.
Specifically,

* data pre-processing leverages the [refer](https://github.com/lichengunc/refer) and [MTTR](https://github.com/mttr2021/MTTR) repository,
* the backbone model is implemented based on code from [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation),
* the training and testing pipelines are adapted from [RefVOS](https://github.com/miriambellver/refvos),
* and implementation of the BERT model (files in the bert directory) is from [Hugging Face Transformers v3.0.2](https://github.com/huggingface/transformers/tree/v3.0.2)
  (we migrated over the relevant code to fix a bug and simplify the installation process).

Some of these repositories in turn adapt code from [OpenMMLab](https://github.com/open-mmlab) and [TorchVision](https://github.com/pytorch/vision).
We'd like to thank the authors/organizations of these repositories for open sourcing their projects.




