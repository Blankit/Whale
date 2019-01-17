from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, cls, label = sample['image'],sample['class'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img, 'class': cls, 'label': label}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, cls, label = sample['image'], sample['class'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'class': cls, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, cls, label = sample['image'], sample['class'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print("*"*10,type(image))
        # print("*"*10,image.shape)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'class':cls,
                'label': label}#torch.Tensor(label)

class WhaleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir,cls_csv, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            #########
            cls_csv(string):类别与标签对应的表
        """
        self.img = pd.read_csv(csv_file)
        self.cls_label = pd.read_csv(cls_csv)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.img.iloc[idx, 0])
        img_label = self.img.iloc[idx, 1]
        label = self.cls_label[self.cls_label.cls == img_label].index.tolist()[0]
        image = io.imread(img_name)
        # landmarks = self.img.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'class':img_label,'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image)
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated



# Helper function to show a batch
def show_whale_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch = \
            sample_batched['image'], sample_batched['class']
    batch_size = len(images_batch)
    # print('batch_size:',batch_size)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(label_batch)
def main():
    transformed_dataset = WhaleDataset(csv_file='results.csv',
                                       root_dir='dataset/',
                                       cls_csv='class_label.csv',
                                       transform=transforms.Compose([
                                           Rescale(256),
                                           RandomCrop(224),
                                           ToTensor()]))

    dataloader = DataLoader(transformed_dataset, batch_size=2,
                            shuffle=True, num_workers=1)

    # for i_batch, sample_batched in enumerate(dataloader):
    #
    #     print(i_batch, sample_batched['image'].size(),sample_batched['label'],
    #           sample_batched['class'])
    #     plt.figure()
    #     # print('len_sample_batched',len(sample_batched))
    #     show_whale_batch(sample_batched)
    #     plt.axis('off')
    #     plt.ioff()
    #     plt.show()



    for i_batch, sample_batched in enumerate(dataloader):
        pass

        # observe 4th batch and stop.
        # if i_batch == 3:
        #     plt.figure()
        #     show_whale_batch(sample_batched)
        #     plt.axis('off')
        #     plt.ioff()
        #     plt.show()
        #     break


if __name__ == '__main__':
    main()