import os
import random
from PIL import Image
from image_captioning import image_to_text

# def load_image_name():
#     file_list = os.listdir('./flickr30k-images')
#     file_name = []
#     for file in file_list:
#         name,_ = file.split('.')
#         file_name.append(name)
#     print(len(file_name),'images are!')
#     return file_name

# def sample_image(sample , n = 1):
#     image = random.sample(sample,n)
#     return image

# def load_image(images):
#     image_list = []
#     for i in images:
#         image = Image.open('./flickr30k-images/'+str(i)+'.jpg')
#         image_list.append(image)
#     return image_list
        
# def load_reference_sentences(images):
#     reference_list = []
#     for i in images:
#         reference = []
#         with open('./flick30k/results_20130124.token','r') as f:
#             for line in f:
#                 if str(line.split()[0].split('.')[0]) == i:
#                     reference.append(line.split("\t")[1])
#         reference_list.append(reference)
#     return reference_list

# def image_sentence(n = 1):      
#     names = load_image_name()
#     images = sample_image(names,n=n)
#     i = load_image(images=images)
#     s = load_reference_sentences(images=images)
#     return i,s

# def image_truncation(images,center = [50,50],hor = 10,ver = 10):
#     croppedImages = []
#     for image in images:
#         horizontal,vertical = image.size
#         croppedImage = image.crop((center[0]-hor,center[1]-ver,center[0]+hor,center[1]+ver))
#         croppedImages.append(croppedImage)
#     return croppedImages

# i,s = image_sentence(n=1)
# croppoedImages = image_truncation(i,center = [100,100],hor=100,ver=100)
# croppoedImages[0].show()



#######
def load_image():
    image = Image.open('./ex/ex1.jpg')
    return image

def image_truncation(image,center = [50,50],hor = 10,ver = 10):
    horizontal,vertical = image.size
    croppedImage = image.crop((center[0]-hor,center[1]-ver,center[0]+hor,center[1]+ver))
    return croppedImage

i = load_image()
print(i.size)
croppoedImages = image_truncation(i,center = [1750,780],hor=150,ver=100) #ex1
# croppoedImages = image_truncation(i,center = [100,300],hor=100,ver=100) #ex2
# croppoedImages = image_truncation(i,center = [600,600],hor=600,ver=100) #ex
# croppoedImages = image_truncation(i,center = [300,250],hor=300,ver=130) #ex3
Image_to_text = image_to_text()
raw = Image_to_text.image_captioning(i)
cut = Image_to_text.image_captioning(croppoedImages)
print(raw[0]['generated_text'],cut[0]['generated_text'])
croppoedImages.show()