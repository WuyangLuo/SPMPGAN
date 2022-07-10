# Context-Consistent Semantic Image Editing with Style-Preserved Modulation

Wuyang Luo, Su Yang, Hong Wang, Bo Long, Weishan Zhang

[Paper (ECCV 2022)]()

![SPMPGAN teaser](images/apps.jpg)

## Requirements

- The code has been tested with PyTorch 1.10.1 and Python 3.7.11. We train our model with a NIVIDA RTX3090 GPU.

## Training

### Dataset Preparation
Download [Cityscapes](https://www.cityscapes-dataset.com/) or [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip). Create folder `data/dataset_name/` with subfolders `train/` and `tes/t`. `train/` and `test/` should each have their own subfolders `images/`, `labels/`, `inst_map/`.
- `images/`: Original images.
- `labels/`: Segmentation maps.
- `inst_map/`(optional): Instance maps for generating edge maps. We find edge maps only have a slight impact. If there is no instance map, it can be omitted or replaced with segmentation map.

Train a model:
```python 
train.py --dataset_name cityscapes
```

## Testing

Download pretrained model from [BaiDuYun (password:z6jz)](https://pan.baidu.com/s/1u4QZALqPjPTvJ5Fr9UIGAQ) | [GoogleDrive](https://drive.google.com/file/d/17FXdCFWx44NiBGW6erM-cJzcW1GpvM3l/view?usp=sharing), run
```python 
test.py --dataset_name cityscapes  --ckt_path pretrained_models/cityscapes.pth --image_path data_test/input.jpg --segmap_path data_test/segmap_1.png --mask_path  data_test/mask_1.png
```

## BibTex:
```

```

## Acknowledgment
Our code is developed based on [SPADE](https://github.com/NVlabs/SPADE).
