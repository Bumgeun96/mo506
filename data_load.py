from PIL import Image
import json

EX = './ex/ex.jpg'
EX1 = './ex/ex1.jpg'
EX2 = './ex/ex2.jpg'
EX3 = './ex/ex3.webp'

def load_image_data(image = './ex/ex1.jpg'):
    image_name = image.split('/')[-1]
    with open("./ex/det_examples.json") as f:
        json_data = json.load(f)
    for i in json_data['frames']:
        if i['name']==image_name:
            json_ = i
            break
    image = Image.open(image)
    return image,json_

def get_center(json_):
    for i in json_['labels']:
        if i['score'] > 0.6:
            x = (i['box2d']['x1']+i['box2d']['x2'])/2
            y = (i['box2d']['y1']+i['box2d']['y2'])/2
            x1 = i['box2d']['x1']
            x2 = i['box2d']['x2']
            y1 = i['box2d']['y1']
            y2 = i['box2d']['y2']
            a = i['score']
    return [x,y,x1,x2,y1,y2,a]
            
i,j = load_image_data(image=EX1)
get_center(j)          
            
def get_image_and_center():
    i,j = load_image_data(image=EX1)
    c = get_center(j)
    return i,c
