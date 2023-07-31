# %%
import numpy as np
import os
import torch
import pandas as pd
import torchvision.datasets as datasets
import tifffile
import cv2
from sklearn.metrics import classification_report, balanced_accuracy_score
import shutil
import matplotlib.pyplot as plt
import random

# # %% read sensitivity
# from sklearn.metrics import roc_curve
# csv = '/home/pwuaj/hkust/DR/test.csv'
# csv = pd.read_csv(csv)
# predict = np.array(csv['Probability'])
# # plt.hist(predict, bins = 50)
# label = np.array(csv['True label'])
# fpr, tpr, thresholds = roc_curve(label, predict, drop_intermediate = False)
# th = thresholds[np.argmax(tpr-fpr)]
# predict = (predict >= th)*1
# print(th,classification_report(label, predict, digits = 4))


# # %% UWF data
# csv = '/home/pwuaj/hkust/DR/gradabilityprobs_labels.csv'
# csv = pd.read_csv(csv)
# csv
# folder = '/home/pwuaj/data/UWF'
# roots = []
# files = []
# for root, dir, file in os.walk(folder):
#    for name in file:
#       roots.append(root)
#       files.append(name)
# for i in range(len(csv)):
#     name = csv['Image'].loc[i]
#     lbl = int(csv['Label (gradable = 0, ungradable =1)'].loc[i])
#     type = csv['Split'].loc[i]
#     try:
#         shutil.copyfile(os.path.join(*[roots[files.index(name)], name]), os.path.join(*['/home/pwuaj/data/grad', type, str(lbl), name]))
#     except:
#         print(name)


# # %% RDR VTDR
# a = ['RDR', 'VTDR']
# for i in a:
#     root = os.path.join(*['/home/pwuaj/data/IS/gradable', i])
#     classes = ['0', '1']
#     for c in classes:
#         files = os.listdir(os.path.join(*[root, c]))
#         train = round(len(files)*12/25)
#         valid = round(len(files)*1/25)
#         test = round(len(files)*12/25)
#         random.shuffle(files)
#         trainlist = files[:train]
#         validlist = files[train:train+valid]
#         testlist = files[train+valid:]
#         for l in trainlist:
#             shutil.copyfile(os.path.join(*[root, c, l]), os.path.join(*['/home/pwuaj/data', i, 'training', c, l]))
#         for l in validlist:
#             shutil.copyfile(os.path.join(*[root, c, l]), os.path.join(*['/home/pwuaj/data', i, 'validation', c, l]))
#         for l in testlist:
#             shutil.copyfile(os.path.join(*[root, c, l]), os.path.join(*['/home/pwuaj/data/IStest', i, c, l]))
#             shutil.copyfile(os.path.join(*[root, c, l]), os.path.join(*['/home/pwuaj/data', i, 'test', c, l]))
#
# a = ['RDR', 'VTDR']
# for i in a:
#     root = os.path.join(*['/home/pwuaj/data/GEI/gradable', i])
#     classes = ['0', '1']
#     for c in classes:
#         files = os.listdir(os.path.join(*[root, c]))
#         train = round(len(files)*16/25)
#         valid = round(len(files)*4/25)
#         test = round(len(files)*1/5)
#         random.shuffle(files)
#         trainlist = files[:train]
#         validlist = files[train:train+valid]
#         testlist = files[train+valid:]
#         for l in trainlist:
#             shutil.copyfile(os.path.join(*[root, c, l]), os.path.join(*['/home/pwuaj/data', i, 'training', c, l]))
#         for l in validlist:
#             shutil.copyfile(os.path.join(*[root, c, l]), os.path.join(*['/home/pwuaj/data', i, 'validation', c, l]))
#         for l in testlist:
#             shutil.copyfile(os.path.join(*[root, c, l]), os.path.join(*['/home/pwuaj/data/GEItest', i, c, l]))
#             shutil.copyfile(os.path.join(*[root, c, l]), os.path.join(*['/home/pwuaj/data', i, 'test', c, l]))

# %%
SK = '/home/pwuaj/data/SK'
GEI = '/home/pwuaj/data/GEI'
IS = '/home/pwuaj/data/IS'
dir = [SK, GEI, IS]
types = ['0', '1']
output = '/home/pwuaj/data/grad'
for i in dir:
    gradable = os.path.join(i, 'gradable/RDR')
    ungradable = os.path.join(i, 'ungradable')
    for name in os.listdir(ungradable):
        print(output)
        shutil.copyfile(os.path.join(ungradable, name), os.path.join(*[output, i, str(0), name]))
    for type in types:
        for name in os.listdir(os.path.join(*[gradable, type])):
            shutil.copyfile(os.path.join(*[gradable, type, name]), os.path.join(*[output, i, str(1), name]))

torch.cuda.empty_cache()
