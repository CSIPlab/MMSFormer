<div align="center"> 

## MMSFormer: Multimodal Material Segmentation with Multimodal Segmentation Transformer

</div>

<!-- <p align="center">
<a href="https://arxiv.org/pdf/2303.01480.pdf">
    <img src="https://img.shields.io/badge/arXiv-2303.01480-red" /></a>
<a href="https://jamycheung.github.io/DELIVER.html">
    <img src="https://img.shields.io/badge/Project-page-green" /></a>
<a href="https://www.youtube.com/watch?v=X-VeSLsEToA">
    <img src="https://img.shields.io/badge/Video-YouTube-%23FF0000.svg" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" /></a>
<a href="https://github.com/jamycheung/DELIVER/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
</p> -->

## Introduction

Leveraging information across diverse modalities is known to enhance performance on segmentation tasks. However, effectively fusing information from different modalities remains challenging due to the unique characteristics of each modality. In this paper, we propose a novel fusion strategy that can effectively fuse information from four different modalities: RGB, AoLP, DoLP and NIR. We also propose a new model named **M**ulti-**M**odal **S**egmentation Trans**Former** **(MMSFormer)** that incorporates the proposed fusion strategy to perform multimodal material segmentation task. MMSFormer achieves **52.05\% mIoU** outperforming the current state-of-the-art by 0.51\% on MCubeS dataset. Starting from RGB-only, performance gradually increases as we add new modalities. It shows significant improvement in detecting **gravel (+10.4\%)** and **human(+9.1\%)** classes. Ablation studies show that different modules in the fusion block are crucial for overall model performance. Our code and pretrained models are publicly available for reference. 

For more details, please check our [arXiv]() paper.

## Updates
- [x] 08/2023, init repository.
- [x] 08/2023, release the code for MMSFormer.
- [x] 08/2023, release MMSFormer model weights. Download from [**GoogleDrive**](https://drive.google.com/drive/folders/1gYciyPj5VvE1AJcuYA8JGmWh61OF3asH?usp=sharing).

## MMSFormer model

<div align="center"> 

![MMSFormer](figs/MMSFormer-Overall.png)
**Figure:** Overall architecture of MMSFormer model.

![Fusion Block](figs/MMSFormer-Fusion.png)
**Figure:** Proposed multimodal fusion block. 
</div>

## Environment

First, create and activate the environment using the following commands: 
```bash
conda env create -f environment.yaml
conda activate mmsformer
# Optional: install apex follow: https://github.com/NVIDIA/apex
```

## Data preparation
Download the dataset:
- [MCubeS](https://github.com/kyotovision-public/multimodal-material-segmentation), for multimodal material segmentation with RGB-A-D-N modalities.

Then, put the dataset under `data` directory as follows:

```
data/
├── MCubeS
│   ├── polL_color
│   ├── polL_aolp
│   ├── polL_dolp
│   ├── NIR_warped
│   └── SS
```

## Model Zoo

### MCubeS
| Model-Modal      | mIoU   | weight |
| :--------------- | :----- | :----- |
| MMSFormer-RGB       | 50.07 | [GoogleDrive](https://drive.google.com/drive/folders/18WXcJxfJsK_0UzKTYENdQaaEFWDo6xW6?usp=sharing) |
| MMSFormer-RGB-A     | 51.28 | [GoogleDrive](https://drive.google.com/drive/folders/18WXcJxfJsK_0UzKTYENdQaaEFWDo6xW6?usp=sharing) |
| MMSFormer-RGB-A-D   | 51.57 | [GoogleDrive](https://drive.google.com/drive/folders/18WXcJxfJsK_0UzKTYENdQaaEFWDo6xW6?usp=sharing) |
| MMSFormer-RGB-A-D-N | 52.05 | [GoogleDrive](https://drive.google.com/drive/folders/18WXcJxfJsK_0UzKTYENdQaaEFWDo6xW6?usp=sharing) |


## Training

Before training, please download [pre-trained SegFormer](https://drive.google.com/drive/folders/1Gx0DCwfsyoRs1pHAS6KksoGBdZEHxcE3?usp=sharing), and put it in the correct directory following this structure:

```text
checkpoints/pretrained/segformer
├── mit_b2.pth
└── mit_b4.pth
```

To train MMSFormer model, please change the `configs/mcubes_rgbadn.yaml` file with appropriate path and hyper-parameters. 

```bash
cd path/to/MMSFormer
conda activate mmsformer

python -m tools.train_mm --cfg configs/mcubes_rgbadn.yaml
```


## Evaluation
To evaluate MMSFormer models, please download respective model weights ([**GoogleDrive**](https://drive.google.com/drive/folders/18WXcJxfJsK_0UzKTYENdQaaEFWDo6xW6?usp=sharing)) as:


```text
output/
├── MCubeS
│   ├── MMSFormer_MiT_B2_MCubeS_RGB.pth
│   ├── MMSFormer_MiT_B2_MCubeS_RGBA.pth
│   ├── MMSFormer_MiT_B2_MCubeS_RGBAD.pth
│   ├── MMSFormer_MiT_B2_MCubeS_RGBNAD.pth
```

Then, modify `configs/mcubes_rgbadn.yaml` file, and run:

```bash
cd path/to/MMSFormer
conda activate mmsformer

python -m tools.val_mm --cfg configs/mcubes_rgbadn.yaml
```

## License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.


## Citations

If you use MMSFormer model, please cite the following works:

- **MMSFormer** [[**PDF**]()]
<!-- ```
@article{zhang2023delivering,
  title={Delivering Arbitrary-Modal Semantic Segmentation},
  author={Zhang, Jiaming and Liu, Ruiping and Shi, Hao and Yang, Kailun and Reiß, Simon and Peng, Kunyu and Fu, Haodong and Wang, Kaiwei and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2303.01480},
  year={2023}
}
``` -->

## Acknowledgements
Our codebase is based on the following Github repositories. Thanks for the public repositories:
- [DELIVER](https://github.com/jamycheung/DELIVER)
- [RGBX-semantic-segmentation](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)
- [Semantic-segmentation](https://github.com/sithu31296/semantic-segmentation)

**Note:** This is a research level repository and might contain issues/bugs. Please contact the authors for any query.