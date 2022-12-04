# mo506 @KAIST
filckr30k image: P. Young, A. Lai, M. Hodosh, and J. Hockenmaier. From image description to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics (to appear).
dataset: https://paperswithcode.com/dataset/flickr30k

# Image Captioning
```
https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
https://cocodataset.org/#captions-2015
https://vizwiz.org/tasks-and-datasets/image-captioning/
```

# flick30k
## Download the dataset for image caption
```
https://kaistackr-my.sharepoint.com/:u:/g/personal/j4t123_kaist_ac_kr/EcHnK2WyP_5Emj9SwbFX7FMBQyWSX47Gl8MNTfYEfWdxbg?e=1xi1mQ
```
## Inference
Json
```
https://kaistackr-my.sharepoint.com/:u:/g/personal/jihui_kaist_ac_kr/EXP58v0qhP1Iv3Hb9auCXf0Bh_9_9PwP21aW3_JJXsiUGQ?e=IiDIMh
```
# BDD100K
## Download the dataset for object detection
```
  wget https://kaistackr-my.sharepoint.com/:u:/g/personal/jihui_kaist_ac_kr/EViLj7RRZk9MvFT4LyLTvyMBAXR-mu_ceONE-xjzjtGIBA?e=D3xhxG
```
trees
```
  └── bdd100k
    ├── images
    │   └── 100k
    │       ├── test
    │       ├── train
    │       └── val
    ├── jsons
    └── labels
        └── det_20
```
## Results
will be updated
```
  https://kaistackr-my.sharepoint.com/:u:/g/personal/jihui_kaist_ac_kr/EQ-Y69KrUp9Eno6nxfhOjFABALuiWYrxfPAQxn-SZhOykw?e=Offh4m
```
### Example
Download faster_rcnn_r50_fpn_1x_det_bdd100k.json
```
  wget https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r50_fpn_1x_det_bdd100k.json
```
### json file structure
```
{
  "frames": [
    {
      "name": "img_name.jpg",
      "labels": [
        {
          "id": "1", //order
          "score": 0.47877612709999084,
          "category": "pedestrian",
          "box2d": {
            "x1": 651.2647705078125,
            "y1": 352.7582092285156,
            "x2": 662.819580078125,
            "y2": 386.6582946777344
          }
        },
...
}
```
