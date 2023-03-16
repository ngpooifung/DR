# %%
import numpy as np
import os
import pandas as pd
import torchvision.datasets as datasets
import tifffile
import cv2

# %%
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
# img_dir = './UWF'
# output_dir = './gradable'
# ungrad_dir = './ungradable'
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
#         if img.split('.')[0] in list(gradable['IMG']):
#             image = cv2.imread(os.path.join(*[dir, img]))
#             index = gradable.index[gradable['IMG'] == img.split('.')[0]]
#             if len(index)>1:
#                 print(index)
#                 continue
#             for cla in classes:
#                 label = gradable[cla][index.item()]
#                 tifffile.imsave(os.path.join(*[output_dir, cla, str(label), img.split('.')[0] + '.tif']), image)
#
#         elif img.split('.')[0] in list(ungradable['IMG']):
#             image = cv2.imread(os.path.join(*[dir, img]))
#             tifffile.imsave(os.path.join(*[ungrad_dir, img.split('.')[0] + '.tif']), image)

# %%
jpg_dir = '/scratch/PI/eeaaquadeer/Phoom/RDR'
tiff_dir = '/scratch/PI/eeaaquadeer/Phoom/RDRt'
split = ['test', 'training', 'validation']
classes = ['0', '1']
for sp in split:
    for cla in classes:
        imglist = os.listdir(os.path.join(*[jpg_dir, sp, cla]))
        for img in imglist:
            image = cv2.imread(os.path.join(*[jpg_dir, sp, cla, img]))
            tifffile.imsave(os.path.join(*[tiff_dir, sp, cla, img.split('.')[0] + '.tif']), image)

# # %% read accuracy
# sp = 'validation'
# csv = 'Phoom/rrecent.csv'
# csv = pd.read_csv(csv)
# csv
# predict = []
# for i in range(sum(csv['Split'] == sp)):
#     predict.append(float(csv['Predicted Probability'][csv['Split'] == sp].iloc[i][1:6]))
# predict = np.array(predict)
# predict = (predict > 0.5)*1
# label = np.array(csv['Label'][csv['Split'] == sp])
#
# sum(predict == label)/sum(csv['Split'] == sp)
