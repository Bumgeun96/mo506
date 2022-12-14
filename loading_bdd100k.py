import json
from PIL import Image
import random

def data_set():
    with open("./faster_rcnn_r50_fpn_1x_det_bdd100k.json") as f:
        json_data = json.load(f)
    return json_data

def data_set1():
    with open("./det0.json") as f:
        json_data = json.load(f)
    return json_data

def sample_data(json_data,n=1):
    sampled_data = random.sample(json_data['frames'],n)
    while sampled_data[0]['labels'] == []:
        sampled_data = random.sample(json_data['frames'],n)
    return sampled_data

def load_image_name(json_data):
    image_name = []
    for i in json_data:
        image_name.append(i["name"])
    return image_name

def load_image(images):
    image_list = []
    for i in images:
        image = Image.open('./bdd100k/images/100k/val/'+str(i))
        image_list.append(image)
    return image_list

def load_image1(images):
    image_list = []
    for i in images:
        image = Image.open('./flickr30k-images/'+str(i))
        image_list.append(image)
    return image_list

def load_reference_sentences(images):
    reference_list = []
    for i in images:
        reference = []
        with open('./flick30k/results_20130124.token','r') as f:
            for line in f:
                if str(line.split()[0].split('#')[0]) == i:
                    reference.append(line.split("\t")[1].split("\n")[0])
        reference_list.append(reference)
    return reference_list

def center_point(data):
    x_y = []
    x_y_a = []
    for d in data:
        x = 0
        y = 0
        x_a = 0
        y_a = 0
        a = 0
        for n,l in enumerate(d["labels"]):
            a += l["score"]
            x1 = l["box2d"]["x1"]
            x2 = l["box2d"]["x2"]
            x += (x1+x2)/2
            y1 = l["box2d"]["y1"]
            y2 = l["box2d"]["y2"]
            y += (y1+y2)/2
            x_a += l["score"]*(x1+x2)/2
            y_a += l["score"]*(y1+y2)/2
        x_y.append([x/(n+1),y/(n+1)])
        x_y_a.append([x_a/a,y_a/a])
        
    center = x_y
    center_attention = x_y_a
    return center,center_attention
        
def get_image_center(n = 1):
    data = data_set()
    sampled_data = sample_data(data,n)
    image_name = load_image_name(sampled_data)
    image = load_image(image_name)
    center,center_attention = center_point(sampled_data)
    return image,center,center_attention

def get_image_center1(n = 1):
    data = data_set1()
    sampled_data = sample_data(data,n)
    image_name = load_image_name(sampled_data)
    reference_sentences = load_reference_sentences(image_name)
    image = load_image1(image_name)
    center,center_attention = center_point(sampled_data)
    return image,reference_sentences,center,center_attention