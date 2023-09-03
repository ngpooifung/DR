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
# predict = (predict >= 0.75)*1
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

# # %%
# SK = '/home/pwuaj/data/SK'
# GEI = '/home/pwuaj/data/GEI'
# IS = '/home/pwuaj/data/IS'
# dir = [SK, GEI, IS]
# types = ['0', '1']
# for i in dir:
#     gradable = os.path.join(i, 'gradable/RDR')
#     ungradable = os.path.join(i, 'ungradable')
#     output = os.path.join(*['/home/pwuaj/data/grad', i.split('/')[-1]])
#     for name in os.listdir(ungradable):
#         shutil.copyfile(os.path.join(*[ungradable, name]), os.path.join(*[output, str(0), name]))
#     for type in types:
#         for name in os.listdir(os.path.join(*[gradable, type])):
#             shutil.copyfile(os.path.join(*[gradable, type, name]), os.path.join(*[output, str(1), name]))
#
# torch.cuda.empty_cache()

# # %%
# csv = '/home/pwuaj/hkust/DR/Original Grading for WF project.xlsx'
# csv = pd.read_excel(csv)
# csv
# gradable = []
# ungradable = []
# name = ['Image name 1', 'Image name 2', 'Image name 3', 'Image name 4', 'Image name 5', 'Image name 6', 'Image name 7']
# grad = ['Gradability 1 (gradable = 1 ungradable = 2)', 'Gradability 2', 'Gradability  3', 'Gradability  4', 'Gradability  5', 'Gradability  6', 'Gradability  7']
# for i in range(len(csv)):
#     for j in range(7):
#         if not pd.isnull(csv.iloc[i][name[j]]):
#             if csv.iloc[i][grad[j]] == 2.0:
#                 ungradable.append(csv.iloc[i][name[j]])
#
# folder = '/home/pwuaj/data/RDRraw'
# roots = []
# gradable = []
# for root, dir, file in os.walk(folder):
#    for name in file:
#       roots.append(root)
#       gradable.append(name)
#
#
# gradable1 = []
# ungradable1 = []
# for i in gradable:
#     if i not in gradable1:
#         gradable1.append(i)
#
# for i in ungradable:
#     if i not in ungradable1:
#         ungradable1.append(i)
# print(len(gradable1), len(ungradable1))
#
# # %%
# folder = '/home/pwuaj/data/UWF'
# roots = []
# gradfiles = []
# for root, dir, file in os.walk(folder):
#    for name in file:
#       roots.append(root)
#       gradfiles.append(name)
#
# for i in ungradable1:
#     j = i + '.jpg'
#     if j not in gradfiles:
#         ungradable1.remove(i)
#
#
# classes = [ungradable1, gradable1]
# for i in range(2):
#     files = classes[i]
#     train = round(len(files)*16/25)
#     valid = round(len(files)*4/25)
#     test = round(len(files)*1/5)
#     random.shuffle(files)
#     trainlist = files[:train]
#     validlist = files[train:train+valid]
#     testlist = files[train+valid:]
#     for l in trainlist:
#         if l[-3:] != 'jpg':
#             l = l + '.jpg'
#         shutil.copyfile(os.path.join(*[roots[gradfiles.index(l)], l]), os.path.join(*['/home/pwuaj/data/grad2', 'training', str(i), l]))
#     for l in validlist:
#         if l[-3:] != 'jpg':
#             l = l + '.jpg'
#         shutil.copyfile(os.path.join(*[roots[gradfiles.index(l)], l]), os.path.join(*['/home/pwuaj/data/grad2', 'validation', str(i), l]))
#     for l in testlist:
#         if l[-3:] != 'jpg':
#             l = l + '.jpg'
#         shutil.copyfile(os.path.join(*[roots[gradfiles.index(l)], l]), os.path.join(*['/home/pwuaj/data/grad2', 'test', str(i), l]))


# %%
csv = '/home/pwuaj/hkust/DR/All_20200715.xlsx'
csv = pd.read_excel(csv, usecols = 'AK, AS, AT, AZ')
csv
folder = '/home/pwuaj/data/GEI'
roots = []
files = []
for root, dir, file in os.walk(folder):
   for name in file:
      roots.append(root)
      files.append(name)

for i in range(217):
    name = csv['image'].iloc[i]
    U = int(csv['Label_U3'].iloc[i])
    if U == 0:
        V = int(csv['Label_V3'].iloc[i])
        R = int(csv['Label_R3'].iloc[i])
        try:
            shutil.copyfile(os.path.join(*[roots[files.index(name)], name]), os.path.join(*['/home/pwuaj/data/GEI2/gradable/RDR', str(R), name]))
        except:
            print(name)
        try:
            shutil.copyfile(os.path.join(*[roots[files.index(name)], name]), os.path.join(*['/home/pwuaj/data/GEI2/gradable/VTDR', str(V), name]))
        except:
            print(name)
    if U == 1:
        shutil.copyfile(os.path.join(*[roots[files.index(name)], name]), os.path.join(*['/home/pwuaj/data/GEI2/ungradable', name]))
