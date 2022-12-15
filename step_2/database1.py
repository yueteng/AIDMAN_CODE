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
import random


class Dataset(object):
    def __init__(self, path, dataset='dataset'):
        self.path = path
        self.dataset = dataset

        pos_path = os.path.join(path, 'Parasitized')
        pos_images = glob.glob(pos_path + '/*')
        neg_path = os.path.join(path, 'Uninfected')
        neg_images = glob.glob(neg_path + '/*')

        X, Y = list(), list()
        X.extend(self.load_images(pos_images))
        Y.extend([1] * len(pos_images))

        X.extend(self.load_images(neg_images))
        Y.extend([0] * len(neg_images))

        X, Y = np.array(X), np.array(Y)
        self.X, self.Y = X, Y,
        self.name = pos_images + neg_images

    def train_valid_split(self, ratio=0.2):
        """ 训练-验证-测试集分割 """
        split_path = f'./data/{self.dataset}/train_valid_test'
        if not os.path.exists(split_path):
            os.mkdir(split_path)
        np.random.seed(2020)
        num = self.X.shape[0]
        indices = np.arange(num)
        np.random.shuffle(indices)

        valid_indices = indices[0:int(ratio*num)]
        test_indices = indices[int(ratio * num): int(ratio * num)*2]
        train_indices = indices[int(ratio*num)*2:]


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


    def test_data(self):
        """ 训练-验证-测试集分割 """
        split_path = f'./data/{self.dataset}/train_valid_test'
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        np.save(os.path.join(split_path, 'test_data.npy'), self.X)
        np.save(os.path.join(split_path, 'test_label.npy'), self.Y)

    def prediction_data(self):
        x = self.X
        y = self.Y
        # X, Y, pic_name  = list(), list(), list()
        # indices = np.arange(x.shape[0])
        # np.random.shuffle(indices)
        #
        # for i in indices:
        #     X.append(x[i])
        #     Y.append(y[i])
        #     pic_name.append(self.name[i]
        #     )
        pic_name = self.name
        return x, y, pic_name


if __name__ == '__main__':
    dataset = 'dataset1'
    path = "data\\data_20220425\\"
    dataset = Dataset(path, dataset)
    print(f"split train/valid dataset ...")
    dataset.train_valid_split()

    # path = "data\\data_20220425\\"
    # dataset = Dataset(path, dataset)
    # print(f"test dataset ...")
    # dataset.test_data()

