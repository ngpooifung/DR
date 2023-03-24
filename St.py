import cv2
import os
import numpy as np
from datetime import datetime
img_w=960 #3872
img_h=768 #3072

import csv
training_images_prefix = '/scratch/PI/eeaaquadeer/UWF'

labels = []

sevs = {}
with open('H.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        level= int(row["VTDR"])
        for visit in range(1,8):
            curr = "Im"+str(visit)
            if row[curr]!="":
                name = row[curr]
            else:
                break
            if row["G"+str(visit)]=="2":
                level = 10
            sevs[name]=int(level)
files = os.listdir(training_images_prefix)
filenames=[]
for name in files:
    filenames.append(name)
filenames = sorted(filenames)
has={}
i=0
dirname='/scratch/PI/eeaaquadeer/sp'
if not os.path.exists(dirname):
    os.mkdir(dirname)
for folder in filenames:
    if os.path.exists(training_images_prefix+"/"+folder+"/REDGREEN/"):
        training_images_path = training_images_prefix+"/"+folder+"/REDGREEN/"
    else:
        training_images_path = training_images_prefix+"/"+folder+"/"
    images = os.listdir(training_images_path)
    imagenames=[]
    for imagename in images:
        imagenames.append(imagename)
    imagenames=sorted(imagenames)
    for name in imagenames:
        if '(' in name or (name[-3:]!="tif" and name[-3:]!="jpg"):
            continue
        img = cv2.imread(training_images_path+"/"+name)
        height, width, channels = img.shape
        if height<500 or width<500 or channels!=3 or img[:,:,0].sum()==img[:,:,1].sum():
            continue


        img = cv2.resize(img, (img_w,img_h))
        i=i+1

        '''
        if i%5!=0:
            path=os.path.join(training_prefix,str(label)+"/"+name)
        else:
            path=os.path.join(validation_prefix,str(label)+"/"+name)

        names=name.split('-')
        patient=names[0].upper()
        side=names[2][0]
        date=names[1].split('@')[0]
        '''


        #if (patient,side,date) not in sevs.keys():
            #continue
        #has[(patient,side,date)]=1


        if i%10==0:
            print(name)
        #print(patient,side,date)

        name = name[:-4]
        if name not in sevs.keys():
            continue
        label=sevs[name]
        dirname="./sp/"+str(label)+"/"
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        path = './sp/'+str(label)+"/"+name+".jpg"
        cv2.imwrite(path,img)
