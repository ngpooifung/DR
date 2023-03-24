import cv2
from skimage.io import imread
import os
import numpy as np

use_autofluorescent_images = False
threshold =1
img_w=960 #3872
img_h=768 #3072

if not os.path.exists('/scratch/PI/eeaaquadeer/split'):
    os.makedirs('/scratch/PI/eeaaquadeer/split')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/training'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/training')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/training/0'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/training/0')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/training/1'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/training/1')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/validation'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/validation')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/validation/0'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/validation/0')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/validation/1'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/validation/1')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/test'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/test')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/test/0'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/test/0')

if not os.path.exists('/scratch/PI/eeaaquadeer/split/test/1'):
    os.makedirs('/scratch/PI/eeaaquadeer/split/test/1')

training_prefix = "/scratch/PI/eeaaquadeer/split/" + "training/"
validation_prefix = "/scratch/PI/eeaaquadeer/split/" + "validation/"

test_prefix = "/scratch/PI/eeaaquadeer/split/" + "test/"


training_images_prefix = "/scratch/PI/eeaaquadeer/sp/"

labels = []

for severity in [0,1]:
    training_images_path= training_images_prefix + str(severity)
    files = os.listdir(training_images_path)
    filenames=[]
    for name in files:
        filenames.append(name)
    filenames = sorted(filenames)
    i=0
    j=0
    ct = len(filenames)
    mk = round(ct*0.8)
    for name in filenames:
        img = cv2.resize(cv2.imread(training_images_path+"/"+name), (img_w,img_h))
        label=int(severity>=threshold)

        i=i+1

        if i%5!=0:
            j=j+1
            if j%5!=0:
                path=os.path.join(training_prefix,str(label)+"/"+name)
            else:
                path=os.path.join(validation_prefix,str(label)+"/"+name)
        else:
            path=os.path.join(test_prefix,str(label)+"/"+name)
            print(name,label)
        cv2.imwrite(path,img)
