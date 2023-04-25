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
#
#
# # %% folder
# img_dir = '/scratch/PI/eeaaquadeer/UWF'
# output_dir = '/scratch/PI/eeaaquadeer/Adam'
# ungrad_dir = '/scratch/PI/eeaaquadeer/Adam/ungradable'
# folder = os.listdir(img_dir)
# classes = ['VTDR', 'RDR']
#
# for name in folder:
#     if os.path.exists(os.path.join(*[img_dir, name, 'REDGREEN'])):
#         dir = os.path.join(*[img_dir, name, 'REDGREEN'])
#     else:
#         dir = os.path.join(*[img_dir, name])
#     imglist = os.listdir(dir)
#     for img in imglist:
#
#         if '(' in img or (img[-3:]!="tif" and img[-3:]!="jpg"):
#             continue
#         image = cv2.imread(os.path.join(*[dir, img]))
#         height, width, channels = image.shape
#         if height<500 or width<500 or channels!=3 or image[:,:,0].sum()==image[:,:,1].sum():
#             continue
#         image = cv2.resize(image, (960, 768))
#
#         if img.split('.')[0] in list(gradable['IMG']):
#             index = gradable.index[gradable['IMG'] == img.split('.')[0]]
#             if len(index)>1:
#                 continue
#             for cla in classes:
#                 label = gradable[cla][index.item()]
#                 tifffile.imsave(os.path.join(*[output_dir, cla, str(label), img.split('.')[0] + '.tif']), image)
#
#         elif img.split('.')[0] in list(ungradable['IMG']):
#             tifffile.imsave(os.path.join(*[ungrad_dir, img.split('.')[0] + '.tif']), image)

# # %% jpd to tiff
# jpg_dir = '/scratch/PI/eeaaquadeer/Phoom/RDR'
# tiff_dir = '/scratch/PI/eeaaquadeer/Phoom/RDRt'
# split = ['test', 'training', 'validation']
# classes = ['0', '1']
# for sp in split:
#     for cla in classes:
#         imglist = os.listdir(os.path.join(*[jpg_dir, sp, cla]))
#         for img in imglist:
#             image = cv2.imread(os.path.join(*[jpg_dir, sp, cla, img]))
#             tifffile.imsave(os.path.join(*[tiff_dir, sp, cla, img.split('.')[0] + '.tif']), image)


# # %% read Phoom accuracy
# sp = 'test'
# csv = '/home/pwuaj/hkust/Phoom/vtdrprobs.csv'
# csv = pd.read_csv(csv)
# predict = []
# for i in range(sum(csv['Split'] == sp)):
#     predict.append(float(csv['Predicted Probability'][csv['Split'] == sp].iloc[i][1:5]))
# predict = np.array(predict)
# label = np.array(csv['Label'][csv['Split'] == sp])
# predict = (predict > 0.29)*1
# print(classification_report(label, predict, digits = 4))
#
#
# # %% read excel accuracy
# csv = '/home/pwuaj/hkust/DR/All_20200514.xlsx'
# csv = pd.read_excel(csv)
# predict = []
# label = []
# for i in range(572):
#     predict.append(csv['Pred_R0'][i])
#     label.append(int(csv['Label_R0'][i]))
# predict = np.array(predict)
# label = np.array(label)
# predict = (predict > 0.61755)*1
# print(classification_report(label, predict, digits = 4))
#
# # %%
# threshold = []
# accuracy = []
# for p in np.unique(predict):
#   threshold.append(p)
#   y_pred = (predict >= p).astype(int)
#   accuracy.append(balanced_accuracy_score(label,y_pred))
# threshold[np.argmax(accuracy)]
#
# # %%
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


# # %%
# # read sensitivity
# csv = '/home/pwuaj/hkust/DR/test.csv'
# csv = pd.read_csv(csv)
# predict = np.array(csv['Predicted label'])
# label = np.array(csv['True label'])
# print(classification_report(label, predict, digits = 4))


# %%
csv = '/home/pwuaj/hkust/DR/All_20200715 emma_oneeye.xlsx'
csv = pd.read_excel(csv)
rdr = '/scratch/PI/eeaaquadeer/Phoom/RDRlong/test'
rdrtest = '/scratch/PI/eeaaquadeer/Phoom/RDRtrue2/test'
vtdr = '/scratch/PI/eeaaquadeer/Phoom/VTDRlong/test'
vtdrtest = '/scratch/PI/eeaaquadeer/Phoom/VTDRtrue2/test'
for i in range(len(csv)):
    name = csv['Image_R0 thr_0.66'][i]
    if isinstance(name, str):
        store_lbl = int(csv['Label_R0'][i])
        save_lbl = int(csv['Label_R0'][i])
        try:
            shutil.copyfile(os.path.join(*[rdr, str(store_lbl), name]), os.path.join(*[rdrtest, str(save_lbl), name]))
        except:
            shutil.copyfile(os.path.join(*[rdr, str(1-store_lbl), name]), os.path.join(*[rdrtest, str(save_lbl), name]))
            print(name)

for i in range(len(csv)):
    name = csv['Image_V0'][i]
    if isinstance(name, str):
        store_lbl = int(csv['Label_V0'][i])
        save_lbl = int(csv['Label_V0'][i])
        try:
            shutil.copyfile(os.path.join(*[vtdr, str(store_lbl), name]), os.path.join(*[vtdrtest, str(save_lbl), name]))
        except:
            shutil.copyfile(os.path.join(*[vtdr, str(1-store_lbl), name]), os.path.join(*[vtdrtest, str(save_lbl), name]))
            print(name)
