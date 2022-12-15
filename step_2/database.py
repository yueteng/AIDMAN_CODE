#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/10 16:04
@project: MalariaDetection
@description: 
"""
import glob
import os
import numpy as np
from PIL import Image


class Dataset(object):
    def __init__(self, path, dataset='dataset1'):
        self.path = path
        self.dataset = dataset

        pos_path = os.path.join(path, 'Parasitized')
        pos_images = glob.glob(pos_path + '/*.png')
        neg_path = os.path.join(path, 'Uninfected')
        neg_images = glob.glob(neg_path + '/*.png')

        X, Y = list(), list()
        X.extend(self.load_images(pos_images))
        Y.extend([1] * len(pos_images))

        X.extend(self.load_images(neg_images))
        Y.extend([0] * len(neg_images))

        X, Y = np.array(X), np.array(Y)
        self.X, self.Y = X, Y

    def train_valid_test_split(self, ratio=0.2):
        """ 训练-验证-测试集分割 """
        split_path = f'./data/{self.dataset}/train_valid_test'
        if not os.path.exists(split_path):
            os.mkdir(split_path)
        np.random.seed(2020)
        num = self.X.shape[0]
        indices = np.arange(num)
        np.random.shuffle(indices)

        test_indices = indices[:int(ratio*num)]
        valid_indices = indices[int(ratio*num):int(2*ratio*num)]
        train_indices = indices[int(2*ratio*num):]

        np.save(os.path.join(split_path, 'train_data.npy'), self.X[train_indices])
        np.save(os.path.join(split_path, 'train_label.npy'), self.Y[train_indices])
        np.save(os.path.join(split_path, 'valid_data.npy'), self.X[valid_indices])
        np.save(os.path.join(split_path, 'valid_label.npy'), self.Y[valid_indices])
        np.save(os.path.join(split_path, 'test_data.npy'), self.X[test_indices])
        np.save(os.path.join(split_path, 'test_label.npy'), self.Y[test_indices])

    def cross_fold_split(self, ratio=0.2):
        """ 交叉验证 """
        split_path = f'./data/{self.dataset}/cross_fold'
        if not os.path.exists(split_path):
            os.mkdir(split_path)
        np.random.seed(2020)
        num = self.X.shape[0]
        indices = np.arange(num)
        np.random.shuffle(indices)

        split_num = int(1.0/ratio)
        sub_num = int(num*ratio)
        for i in range(split_num):
            sub_indices = indices[i*sub_num:(i+1)*sub_num]
            np.save(os.path.join(split_path, f'data_{i}.npy'), self.X[sub_indices])
            np.save(os.path.join(split_path, f'label_{i}.npy'), self.Y[sub_indices])

    def load_images(self, images_list):
        data = list()
        for image_path in images_list:
            img = Image.open(image_path)
            resized_img = img.resize((64, 64))
            img_array = np.array(resized_img)
            data.append(img_array)
        return data


if __name__ == '__main__':
    dataset = 'public'  # 'dataset1'
    path = "D:\YS\M_D\M_D_dataset"
    dataset = Dataset(path, dataset)

    print(f"split train/valid/test dataset ...")
    dataset.train_valid_test_split()
    print(f"cross fold split ...")
    dataset.cross_fold_split()
