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
python tools/train.py configs_ltdnet/ltdnet/LTDNet_AI_TODv2.py
```
##### Train LEVIR-Ship dataset:
```
python tools/train.py configs_ltdnet/ltdnet/LTDNet_LEVIR_Ship.py
```
The Pretrained models is on the folder work_dirs.

### 3. Checkpoint Download（checkpoint-299.pth）：
Baidu Pan：[https://pan.baidu.com/s/1eyJiSV12hX6gggiuq8-DFA?pwd=uon2](https://pan.baidu.com/s/1Idj3iAB9bmfNqYtXcmCIeA?pwd=t843) 
code：t843  


### 4. Citation

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

    @ARTICLE{10462223,
      author={Liu, Dongyang and Zhang, Junping and Qi, Yunxiao and Wu, Yinhu and Zhang, Ye},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Tiny Object Detection in Remote Sensing Images Based on Object Reconstruction and Multiple Receptive Field Adaptive Feature Enhancement}, 
      year={2024},
      volume={62},
      number={},
      pages={1-13},
      doi={10.1109/TGRS.2024.3381774}}

    @ARTICLE{10462223,
      author={Liu, Dongyang and Zhang, Junping and Qi, Yunxiao and Wu, Yinhu and Zhang, Ye},
      journal={IEEE Geoscience and Remote Sensing Letters}, 
      title={A Tiny Object Detection Method Based on Explicit Semantic Guidance for Remote Sensing Images}, 
      year={2024},
      volume={21},
      number={},
      pages={1-5},
      doi={10.1109/LGRS.2024.3374418}}
