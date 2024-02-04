# %%
import numpy as np
import os
import torch
import pandas as pd
import torchvision.datasets as datasets
import tifffile
from dataset import Modeldataset
import cv2
from sklearn.metrics import classification_report
import shutil
import matplotlib.pyplot as plt
import random
from math import sqrt
import matplotlib.pyplot as plt
import torch

# %% read sensitivity
from sklearn.metrics import roc_curve, precision_recall_curve, cohen_kappa_score, f1_score, roc_auc_score
csv = '/home/pwuaj/hkust/DR/test.csv'
csv = pd.read_csv(csv)
predict = np.array(csv['Probability'])
label = np.array(csv['True label'])
fpr, tpr, thresholds = roc_curve(label, predict, drop_intermediate = True)
# pre4, rec4, thresholds2 = precision_recall_curve(label, predict, drop_intermediate = True)
th = thresholds[np.argmax(tpr-fpr)]
predict = (predict > 0.076)*1
print(th,classification_report(label, predict, digits = 4),roc_auc_score(label, predict), f1_score(label, predict))


# %%
plt.figure()
fig, axs = plt.subplots(2,2)

axs[0,0].plot(100*fpr1, 100*tpr1, color = 'b', label = '384*480')
axs[0,0].plot(100*fpr2, 100*tpr2, color = 'g', label = '448*560')
axs[0,0].plot(100*fpr3, 100*tpr3, color = 'r', label = '512*640')
axs[0,0].plot(100*fpr4, 100*tpr4, color = 'c', label = '576*720')
axs[0,0].set_title('Primary test')
axs[0,0].legend()

axs[0,1].plot(100*fpr5, 100*tpr5, color = 'b', label = '384*480')
axs[0,1].plot(100*fpr6, 100*tpr6, color = 'g', label = '448*560')
axs[0,1].plot(100*fpr7, 100*tpr7, color = 'r', label = '512*640')
axs[0,1].plot(100*fpr8, 100*tpr8, color = 'c', label = '576*720')
axs[0,1].set_title('External 2')
axs[0,1].legend()

axs[1,0].plot(100*fpr9, 100*tpr9, color = 'b', label = '384*480')
axs[1,0].plot(100*fpr10, 100*tpr10, color = 'g', label = '448*560')
axs[1,0].plot(100*fpr11, 100*tpr11, color = 'r', label = '512*640')
axs[1,0].plot(100*fpr12, 100*tpr12, color = 'c', label = '576*720')
axs[1,0].set_title('External 3')
axs[1,0].legend()

axs[1,1].plot(100*fpr13, 100*tpr13, color = 'b', label = '384*480')
axs[1,1].plot(100*fpr14, 100*tpr14, color = 'g', label = '448*560')
axs[1,1].plot(100*fpr15, 100*tpr15, color = 'r', label = '512*640')
axs[1,1].plot(100*fpr16, 100*tpr16, color = 'c', label = '576*720')
axs[1,1].set_title('External 4')
axs[1,1].legend()

for ax in axs.flat:
    ax.set(xlabel='1 - Specificity, %', ylabel='Sensitivity')

for ax in axs.flat:
    ax.label_outer()

fig.suptitle('RDR detection performance in different image sizes')
plt.savefig('/home/pwuaj/hkust/DR/RDRsizes.png')

# %%
plt.figure()
plt.plot(100*rec1, 100*pre1, color = 'b', label = 'Primary test')
plt.plot(100*rec2, 100*pre2, color = 'g', label = 'External 2')
plt.plot(100*rec3, 100*pre3, color = 'r', label = 'External 3')
plt.plot(100*rec4, 100*pre4, color = 'c', label = 'External 4')
plt.legend()
plt.xlabel('Recall, %')
plt.ylabel('Precision, %')
plt.title('VTDR Precision-Recall plot')
plt.savefig('/home/pwuaj/hkust/DR/VTDRpr.png')


# %%
from sklearn.metrics import roc_curve, precision_recall_curve, cohen_kappa_score, f1_score, roc_auc_score
csv = '/home/pwuaj/hkust/DR/All_20200715.xlsx'
csv = pd.read_excel(csv, usecols = 'G, H')
csv = csv.dropna()
predict = np.array(csv.iloc[:,0])
label = np.array(csv.iloc[:,1])
fpr, tpr, thresholds = roc_curve(label, predict, drop_intermediate = False)
th = thresholds[np.argmax(tpr-fpr)]
predict = (predict >= 0.4078)*1
print(th,classification_report(label, predict, digits = 4),roc_auc_score(label, predict), f1_score(label, predict))


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



# RDRdir = '/home/pwuaj/data/RDRraw'
# VTDRdir = '/home/pwuaj/data/VTDRraw'
# types = ['training', 'validation', 'test']
# nonVTDRlist = os.listdir(os.path.join(*[VTDRdir, 'training', '0'])) + os.listdir(os.path.join(*[VTDRdir, 'validation', '0'])) + os.listdir(os.path.join(*[VTDRdir, 'test', '0']))
# VTDRlist = os.listdir(os.path.join(*[VTDRdir, 'training', '1'])) + os.listdir(os.path.join(*[VTDRdir, 'validation', '1'])) + os.listdir(os.path.join(*[VTDRdir, 'test', '1']))
# for type in types:
#     mild = []
#     severe = []
#     nonRDRlist = os.listdir(os.path.join(*[RDRdir, type, '0']))
#     RDRlist = os.listdir(os.path.join(*[RDRdir, type, '1']))
#     for i in RDRlist:
#         if i in nonVTDRlist:
#             mild.append(i)
#         elif i in VTDRlist:
#             severe.append(i)
#         else:
#             print(i)
#     for i in nonRDRlist:
#         shutil.copyfile(os.path.join(*[RDRdir, type, '0', i]), os.path.join(*['/home/pwuaj/data/DR', type, '0', i]))
#     for i in mild:
#         shutil.copyfile(os.path.join(*[RDRdir, type, '1', i]), os.path.join(*['/home/pwuaj/data/DR', type, '1', i]))
#     for i in severe:
#         shutil.copyfile(os.path.join(*[RDRdir, type, '1', i]), os.path.join(*['/home/pwuaj/data/DR', type, '2', i]))
