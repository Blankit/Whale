import csv
import os
import pandas as pd
import numpy as np
import torchvision.models as models

# cla_label = pd.read_csv('class_label.csv')
#
# index = cla_label[cla_label.cla == 'w_6bea436'].index.tolist()[0]
# print(index)
# print(type(index))

vgg = models.vgg11(pretrained=False)

