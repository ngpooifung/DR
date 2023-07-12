# %%
import numpy as np
import os
import pandas as pd
import torchvision.datasets as datasets
import tifffile
import cv2
from sklearn.metrics import classification_report, balanced_accuracy_score
import shutil
import matplotlib.pyplot as plt
import random
# # %%
# csv_dir = './H.csv'
# csv = pd.read_csv(csv_dir)
# gradable = []
# ungradable = []
# for i in range(len(csv)):
#     for j in range(1,8):
#         if csv.loc[i][f'G{j}'] == 1:
#             gradable.append(pd.DataFrame({'IMG': csv.loc[i][f'Im{j}'], 'VTDR': csv.loc[i]['VTDR'], 'RDR': csv.loc[i]['Referable DR']}, index = [0]))
#         elif csv.loc[i][f'G{j}'] == 2:
#             ungradable.append(pd.DataFrame({'IMG': csv.loc[i][f'Im{j}'], 'VTDR': csv.loc[i]['VTDR'], 'RDR': csv.loc[i]['Referable DR']}, index = [0]))
#
# gradable = pd.concat(gradable, ignore_index = True)
# ungradable = pd.concat(ungradable, ignore_index = True)
#
# gradable.to_csv('./gradable.csv')
# ungradable.to_csv('./ungradable.csv')


# # %% read Phoom accuracy
# sp = 'test'
# csv = '/home/pwuaj/hkust/Phoom/vtdrprobs.csv'
# csv = pd.read_csv(csv)
# predict = []
# for i in range(sum(csv['Split'] == sp)):
#     predict.append(float(csv['Predicted Probability'][csv['Split'] == sp].iloc[i][1:5]))
# predict = np.array(predict)
# label = np.array(csv['Label'][csv['Split'] == sp])
# predict = (predict >= 0.43)*1
# print(classification_report(label, predict, digits = 4))


# # %% read excel accuracy
# csv = '/home/pwuaj/hkust/DR/All_20200514.xlsx'
# csv = pd.read_excel(csv, usecols = 'AN, AQ')
# csv
# predict = []
# predictnan = []
# label = []
# for i in range(len(csv)): # 397 408
#     try:
#         predictnan.append(int(csv['Pred_R3'][i]))
#         label.append(int(csv['Label_R3'][i]))
#         predict.append(csv['Pred_R3'][i])
#     except:
#         continue
# predict = np.array(predict)
# label = np.array(label)


# # %% Youden
# from sklearn.metrics import roc_curve
# fpr, tpr, thresholds = roc_curve(label, predict, drop_intermediate = False)
# # thresholds[np.argmin(np.abs(fpr+tpr-1))]
# th = thresholds[np.argmax(tpr-fpr)]
# predict = (predict >= th)*1
# print(classification_report(label, predict, digits = 4))

# # %% read Phoom gradtest
# csv = 'skprobs.csv'
# csv = pd.read_csv(csv)
# predict = []
# label = []
# for i in range(len(csv)):
#     try:
#         predict.append(float(csv['VTDR_predict'].iloc[i][1:6]))
#         label.append(csv['VTDR'].iloc[i])
#     except:
#         pass
# predict = np.array(predict)
# predict = (predict > 0.5)*1
# print(classification_report(label, predict, digits = 4))


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


# # %% SK data
# csv = '/home/pwuaj/hkust/DR/All_20200514.xlsx'
# csv = pd.read_excel(csv, usecols = 'AA, AC')
# folder = '/home/pwuaj/data/Clarity_original'
# roots = []
# files = []
# for root, dir, file in os.walk(folder):
#    for name in file:
#       roots.append(root)
#       files.append(name)
# for i in range(1321):
#     name = csv['Image_U1'].iloc[i]
#     ungradable = int(csv['Label_U1'].iloc[i])
#     if ungradable == 1:
#         shutil.copyfile(os.path.join(*[roots[files.index(name)], name]), os.path.join(*['/home/pwuaj/data/Clarity/ungradable', name]))
#     elif ungradable == 0:
#         shutil.copyfile(os.path.join(*[roots[files.index(name)], name]), os.path.join(*['/home/pwuaj/data/Clarity/gradable', name]))

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
folder = '/home/pwuaj/data/ungradable'
name = os.listdir(folder)
data = pd.DataFrame({'Image_Name': name})
data.to_csv('/home/pwuaj/data/image.csv')
