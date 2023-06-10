# %%
import numpy as np
import os
import pandas as pd
import torchvision.datasets as datasets
import tifffile
import cv2
from sklearn.metrics import classification_report, balanced_accuracy_score
import shutil
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
# csv = '/home/pwuaj/hkust/DR/All_20200715 emma_oneeye.xlsx'
# csv = pd.read_excel(csv)
# predict = []
# label = []
# for i in range(408): # 397 408
#     predict.append(csv['Pred_V0'][i])
#     label.append(int(csv['Label_V0'][i]))
# predict = np.array(predict)
# label = np.array(label)
# predict = (predict >= 0.5)*1
# print(classification_report(label, predict, digits = 4))


# # %% Youden
# from sklearn.metrics import roc_curve
# fpr, tpr, thresholds = roc_curve(label, predict, drop_intermediate = False)
# # thresholds[np.argmin(np.abs(fpr+tpr-1))]
# thresholds[np.argmax(tpr-fpr)]


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
# csv = '/home/pwuaj/hkust/DR/test.csv'
# csv = pd.read_csv(csv)
# predict = np.array(csv['Predicted label'])
# label = np.array(csv['True label'])
# print(classification_report(label, predict, digits = 4))


# %% SK data
csv = pd.read_excel('/home/pwuaj/hkust/DR/All_20200514.xlsx', usecols = 'AF,AN,AO,AT')
csv
a = []
for i in range(217):
    name = csv.loc[i]['image']
    Ungradable = int(csv['Label_U3'][i])
    RDR = int(csv.loc[i]['Label_R3'])
    VTDR = int(csv.loc[i]['Label_V3'])
    try:
        if Ungradable == 1:
            shutil.copyfile(os.path.join(*['/home/pwuaj/data/all', name]), os.path.join(*['/home/pwuaj/data','GEI/ungradable', name]))
        elif Ungradable == 0:
            shutil.copyfile(os.path.join(*['/home/pwuaj/data/all', name]), os.path.join(*['/home/pwuaj/data','GEI/gradable/RDR', str(RDR), name]))
            shutil.copyfile(os.path.join(*['/home/pwuaj/data/all', name]), os.path.join(*['/home/pwuaj/data','GEI/gradable/VTDR', str(VTDR), name]))
    except:
        a.append(name)
print(a)
