# <p align=center> Exploring Lightweight Structures for Tiny Object Detection in Remote Sensing Images    (IEEE TGRS 2025) </p>

### 1. Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)


### 2. Install and start ORFENet:

Note that our ORFENet is based on the [MMDetection 2.24.1](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/dyl96/LTDNet.git
cd LTDNet
pip install -r requirements/build.txt
python setup.py develop
```
### Prepare Dataset:
Download [AI-TODv2](https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing) dataset; Download [LEVIR-Ship](https://github.com/WindVChen/LEVIR-Ship) dataset.

### Train and test：
##### Train aitodv2 dataset:
```
python tools/train.py configs_ltdnet/LTDNet_AI_TODv2.py
```
##### Train LEVIR-Ship dataset:
```
python tools/train.py configs_orfenet/orfenet/LTDNet_LEVIR_Ship.py
```
The Pretrained models is on the folder work_dirs.


### 3. Citation

Please cite our paper if you find the work useful:

    @ARTICLE{10988878,
      author={Liu, Dongyang and Zhang, Junping and Qi, Yunxiao and Xi, Yunqiao and Jin, Jing},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Exploring Lightweight Structures for Tiny Object Detection in Remote Sensing Images}, 
      year={2025},
      volume={63},
      number={},
      pages={1-15},
      doi={10.1109/TGRS.2025.3567345}}
