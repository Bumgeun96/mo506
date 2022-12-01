## Python env

### conda
```
  conda create --name openmmlab python=3.8 -y
  conda activate openmmlab
```

### install torch
```
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
```
### install mmdetection
```
conda activate openmmlab

pip install -U openmim  
mim install mmcv-full

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection       
pip install -v -e .
```

## BDD100K model zoo

### install
```
  git clone https://github.com/SysCV/bdd100k-models.git
```
### add files train.py (from mmdet)

### add bdd100kdataset in mmdet
