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
from sklearn.metrics import roc_curve, precision_recall_curve, cohen_kappa_score, f1_score, roc_auc_score,average_precision_score,balanced_accuracy_score

# %%
# csv = '/home/pwuaj/hkust/DR/Grad/All_20200715.xlsx'
# csv = pd.read_excel(csv, usecols = 'BG, BI')
# csv = csv.dropna()
# predict = np.array(csv.iloc[:,0])
# label = np.array(csv.iloc[:,1])
# csv = pd.read_csv('/home/pwuaj/hkust/DR/VTDRtest.csv')
# predict = np.array(csv['model output'])
# label = np.array(csv['True label'])
# fpr, tpr, thresholds = roc_curve(label, predict, drop_intermediate = False)
# th = thresholds[np.argmax(tpr-fpr)]
# predict = (predict > 0.77)*1
# print(th,classification_report(label, predict, digits = 4), roc_auc_score(label, predict))
#
# %% result 0.5 0.35 0.45
# folder = '/home/pwuaj/hkust/DR'
# thresholds = {'384':0.558, '448':0.61, '576':0.63, 'resnet': 0.8, 'dense':0.7, 'inception':0.39, 'Phoom':0.69, 'hp': 0.055}
# results = []
# for model in ['resnet']:
#     result = []
#     for dataset in ['test', 'SK', 'GEI', 'IS']:
#         file = 'VTDR' + dataset +'.csv'
#         csv = pd.read_csv(os.path.join(*[folder, file]))
#         predict = np.array(csv['model output'])
#         label = np.array(csv['True label'])
#         predict = (predict > thresholds[model])*1
#         report = classification_report(label, predict, digits = 4).split()
#         Specificity = report[6]
#         Sensitivity = report[11]
#         Accuracy = report[15]
#         auroc = roc_auc_score(label, predict)
#         f1 = f1_score(label, predict)
#         ck = cohen_kappa_score(label, predict)
#         ap = average_precision_score(label, predict)
#         ba = balanced_accuracy_score(label, predict)
#         result.append(pd.DataFrame({'Specificity': Specificity, 'Sensitivity': Sensitivity, 'Accuracy': Accuracy, 'auroc': auroc, 'f1': f1, 'cohen': ck, 'average precision': ap, 'balanced accuracy':ba}, index = [dataset]))
#     results.append(pd.concat(result, axis = 0))
# test = pd.concat(results, axis = 1)
# test.to_csv('/home/pwuaj/hkust/DR/test.csv')

# %% Plot
# plt.figure()
# fig, axs = plt.subplots(2,2)
#
# axs[0,0].plot(100*fpr1, 100*tpr1, color = 'b', label = '384*480')
# axs[0,0].plot(100*fpr2, 100*tpr2, color = 'g', label = '448*560')
# axs[0,0].plot(100*fpr3, 100*tpr3, color = 'r', label = '512*640')
# axs[0,0].plot(100*fpr4, 100*tpr4, color = 'c', label = '576*720')
# axs[0,0].set_title('Primary test')
# axs[0,0].legend()
#
# axs[0,1].plot(100*fpr5, 100*tpr5, color = 'b', label = '384*480')
# axs[0,1].plot(100*fpr6, 100*tpr6, color = 'g', label = '448*560')
# axs[0,1].plot(100*fpr7, 100*tpr7, color = 'r', label = '512*640')
# axs[0,1].plot(100*fpr8, 100*tpr8, color = 'c', label = '576*720')
# axs[0,1].set_title('External 2')
# axs[0,1].legend()
#
# axs[1,0].plot(100*fpr9, 100*tpr9, color = 'b', label = '384*480')
# axs[1,0].plot(100*fpr10, 100*tpr10, color = 'g', label = '448*560')
# axs[1,0].plot(100*fpr11, 100*tpr11, color = 'r', label = '512*640')
# axs[1,0].plot(100*fpr12, 100*tpr12, color = 'c', label = '576*720')
# axs[1,0].set_title('External 3')
# axs[1,0].legend()
#
# axs[1,1].plot(100*fpr13, 100*tpr13, color = 'b', label = '384*480')
# axs[1,1].plot(100*fpr14, 100*tpr14, color = 'g', label = '448*560')
# axs[1,1].plot(100*fpr15, 100*tpr15, color = 'r', label = '512*640')
# axs[1,1].plot(100*fpr16, 100*tpr16, color = 'c', label = '576*720')
# axs[1,1].set_title('External 4')
# axs[1,1].legend()
#
# for ax in axs.flat:
#     ax.set(xlabel='1 - Specificity, %', ylabel='Sensitivity')
#
# for ax in axs.flat:
#     ax.label_outer()
#
# fig.suptitle('RDR detection performance in different image sizes')
# plt.savefig('/home/pwuaj/hkust/DR/RDRsizes.png')
#
# # %%
# plt.figure()
# plt.plot(100*rec1, 100*pre1, color = 'b', label = 'Primary test')
# plt.plot(100*rec2, 100*pre2, color = 'g', label = 'External 2')
# plt.plot(100*rec3, 100*pre3, color = 'r', label = 'External 3')
# plt.plot(100*rec4, 100*pre4, color = 'c', label = 'External 4')
# plt.legend()
# plt.xlabel('Recall, %')
# plt.ylabel('Precision, %')
# plt.title('VTDR Precision-Recall plot')
# plt.savefig('/home/pwuaj/hkust/DR/VTDRpr.png')


# # %% Phoomgrad
# result = []
# # %%
# dataset = 'IS'
# csv = '/home/pwuaj/hkust/DR/All_20200715.xlsx'
# csv = pd.read_excel(csv, usecols = 'BG, BI')
# csv = csv.dropna()
# predict = np.array(csv.iloc[:,0])
# label = np.array(csv.iloc[:,1])
# fpr, tpr, thresholds = roc_curve(label, predict, drop_intermediate = False)
# th = thresholds[np.argmax(tpr-fpr)]
# predict = (predict >= 0.3784)*1
# report = classification_report(label, predict, digits = 4).split()
# Specificity = report[6]
# Sensitivity = report[11]
# Accuracy = report[15]
# auroc = roc_auc_score(label, predict)
# f1 = f1_score(label, predict)
# ck = cohen_kappa_score(label, predict)
# ap = average_precision_score(label, predict)
# result.append(pd.DataFrame({'Specificity': Specificity, 'Sensitivity': Sensitivity, 'Accuracy': Accuracy, 'auroc': auroc, 'f1': f1, 'cohen': ck, 'average precision': ap}, index = [dataset]))
# # %%
# test = pd.concat(result, axis = 0)
# test.to_csv('/home/pwuaj/hkust/DR/Phoomgrad.csv')

# # %%
# csv = '/home/pwuaj/data/trainLabels.csv'
# csv = pd.read_csv(csv)
# for i in range(len(csv)):
#     name = csv['image'].iloc[i] + '.jpeg'
#     severity = csv['level'].iloc[i]
#     print(name, severity)
#     RDR = int(severity>=2)*1
#     VTDR = int(severity>=3)*1
#     shutil.copy(os.path.join(*['/home/pwuaj/data/kaggle', name]), os.path.join(*['/home/pwuaj/data/fundus/RDR', str(RDR), name]))
#     shutil.copy(os.path.join(*['/home/pwuaj/data/kaggle', name]), os.path.join(*['/home/pwuaj/data/fundus/VTDR', str(VTDR), name]))

# %%
for i in ['0', '1']:
    folder = os.path.join('/home/pwuaj/data/fundus/VTDR', i)
    filelist = os.listdir(folder)
    random.shuffle(filelist)
    # if i == '0':
    #     for j in range(126):
    #         shutil.copy(os.path.join(folder, filelist[j]), os.path.join(*['/home/pwuaj/data/VTDRraw/training2/0', filelist[j]]))
    if i == '1':
        for j in range(141):
            shutil.copy(os.path.join(folder, filelist[j]), os.path.join(*['/home/pwuaj/data/VTDRraw/training2/1', filelist[j]]))
