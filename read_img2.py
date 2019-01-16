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

class WhaleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.img.iloc[idx, 0])
        img_label = self.img.iloc[idx, 1]
        image = io.imread(img_name)
        # landmarks = self.img.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': img_label}

        if self.transform:
            sample = self.transform(sample)

        return sample

whale_dataset = WhaleDataset(csv_file='results.csv',root_dir='dataset/')
fig = plt.figure()

def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image)
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


for i in range(len(whale_dataset)):
    sample = whale_dataset[i]
    # print(len(sample[0]))

    print(i, sample['image'].shape, sample['label'])
    #
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_image(sample['image'])
    #
    if i == 3:
        plt.show()
        break