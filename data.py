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
# predict = (predict >= 0.17)*1
# print(th,classification_report(label, predict, digits = 4))


# # %%
# from sklearn.metrics import roc_curve
# csv = '/home/pwuaj/hkust/DR/All_20200715.xlsx'
# csv = pd.read_excel(csv, usecols = 'AF, AG')
# csv = csv.dropna()
# predict = np.array(csv.iloc[:,0])
# label = np.array(csv.iloc[:,1])
# fpr, tpr, thresholds = roc_curve(label, predict, drop_intermediate = False)
# th = thresholds[np.argmax(tpr-fpr)]
# predict = (predict >= th)*1
# print(th,classification_report(label, predict, digits = 4))



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


# # %%
# label = pd.read_csv('/home/pwuaj/hkust/DR/Gradability.csv')
# label
# # %%
# namelist = list(label['External 4'].dropna())
# result = pd.read_csv('/home/pwuaj/hkust/DR/gradIS.csv')
# for i in range(len(result)):
#     name = result['Path'].iloc[i].split('/')[-1]
#     prob = result['Probability'].iloc[i]
#     predict = (prob>0.115)*1
#     label['Probability.4'][namelist.index(name)] = prob
#     label['Predicted Label.4'][namelist.index(name)] = predict
#
# # %%
# label.to_csv('/home/pwuaj/hkust/DR/Gradability2.csv')


RDRdir = '/home/pwuaj/data/RDRraw'
VTDRdir = '/home/pwuaj/data/VTDRraw'
types = ['training', 'validation', 'test']
for type in types:
    mild = []
    severe = []
    nonRDRlsit = os.listdir(os.path.join(*[RDRdir, type, '0']))
    RDRlist = os.listdir(os.path.join(*[RDRdir, type, '1']))
    nonVTDRlist = os.listdir(os.path.join(*[VTDRdir, type, '0']))
    VTDRlist = os.listdir(os.path.join(*[VTDRdir, type, '1']))
    for i in RDRlist:
        if i in nonVTDRlist:
            mild.append(i)
        elif i in VTDRlist:
            severe.append(i)
        else:
            print(i)
    for i in nonRDRlist:
        shutil.copyfile(os.path.join(*[RDRdir, type, '0', i]), os.path.join(*['/home/pwuaj/data/DR', type, '0']))
    for i in mild:
        shutil.copyfile(os.path.join(*[RDRdir, type, '1', i]), os.path.join(*['/home/pwuaj/data/DR', type, '1']))
    for i in severe:
        shutil.copyfile(os.path.join(*[RDRdir, type, '1', i]), os.path.join(*['/home/pwuaj/data/DR', type, '2']))
