import csv
import os
import pandas as pd


csv_file='results.csv'
root_dir='dataset/'
cla_label = pd.read_csv('calss_to_label.csv')
cla_label.columns=['class','label']
# print(cla_label)
img = pd.read_csv(csv_file)

img_name = os.path.join(root_dir,img.iloc[0, 0])
img_label = img.iloc[0, 1]
print(img_label)
print(type(img_label))
print('*'*20)
# print(cla_label[])
# label = cla_label.loc[cla_label['class'].isin(['new_whale']),['label']]
label = cla_label.loc[cla_label['class'].isin(['new_whale']),'label']
print(label.index)
print(int(label.index))
print(type(label.index))
label = cla_label.iloc[label.index,1]
print('**********label*******')
print(label)
print(type(label))
# print(type(cla_label.iloc[label.index,1]))
# print(type(label))
# print(img_name,img_label)
# print(class_label[class_label.label == img_label])

