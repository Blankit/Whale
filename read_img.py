from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd

n = 0

img = pd.read_csv('results.csv')
img_name = img.iloc[n,0]
img_label = img.iloc[n,1]
print(img_name,img_label)

def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image)
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
image = io.imread(os.path.join('dataset/', img_name))
print(type(image))
print(image.shape)
show_image(image)#根据图像名，读出图像
plt.show()