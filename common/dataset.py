'''
@file: dataset.py
@version: v1.0
@date: 2018-01-06
@author: ruanxiaoyi
@brief: Pytorch dataset for Tiny-ImageNet-200
@remark: {when} {email} {do what}
'''

import os
from os.path import abspath
from os.path import join as pjoin
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class TIN200Data(data.Dataset):
    '''
    Pytorch dataset for Tiny-ImageNet-200

    ### Parameters:
    - root: The root path of dataset
    - label_map: The path of file 'wnids.txt'
    - data_dir: The data you want to load which must be 'train', 'val' or 'test'
    - loader: Load a image from a given path
    - transform: Transfrom a image to specified format
    - t_transform: Transfrom a label to specified format
    '''

    def __init__(self, root, label_map, data_dir='train',
                 loader=lambda path: Image.open(path).convert('RGB'),
                 transform=transforms.ToTensor()):
        assert data_dir in ('train', 'val', 'test'), "data_dir must be 'train', 'val' or 'test'"
        self.imgs = []
        self.labels = []
        self.data_dir = data_dir
        self.loader = loader
        self.transform = transform

        with open(label_map, 'r') as f:  # wnids.txt
            content = [x.strip() for x in f.readlines()]
            # map string classid to 0-199 interger label
            self.label_map = {classid: index for index, classid in enumerate(content)}

        if data_dir == 'train':
            for path, _, files in os.walk(pjoin(root, 'train')):
                if len(files) == 1000:
                    self.imgs += [abspath(pjoin(path, f)) for f in files]
                    self.labels += [f.split('_')[0] for f in files]
            self.labels = [self.label_map[label] for label in self.labels]

        elif data_dir == 'val':
            with open(pjoin(root, 'val', 'val_annotations.txt')) as f:
                name_map = [x.strip() for x in f.readlines()]
                name_map = [x.split('\t')[:2] for x in name_map]
                name_map = dict(name_map)

            for path, _, files in os.walk(pjoin(root, 'val')):
                if len(files) > 1:
                    self.imgs += [abspath(pjoin(path, f)) for f in files]
                    self.labels += [self.label_map[name_map[f]] for f in files]

        else:
            for path, _, files in os.walk(pjoin(root, 'test')):
                # how to check a sequence(list, tuple, dict) is empty or not with a pythonic way
                # https://stackoverflow.com/questions/43121340/why-is-the-use-of-lensequence-in-condition-values-considered-incorrect-by-pyli
                if files:
                    self.imgs += [abspath(pjoin(path, f)) for f in files]

    def __getitem__(self, index):
        if self.data_dir == 'test':
            img = self.imgs[index]
            img = self.transform(self.loader(img))
            return img

        img = self.imgs[index]
        img = self.transform(self.loader(img))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.imgs)
