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

    def test_data(self, ratio=0.2):
        """ 训练-验证-测试集分割 """
        split_path = f'./data/{self.dataset}/train_valid_test'
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        np.save(os.path.join(split_path, 'test_data.npy'), self.X)
        np.save(os.path.join(split_path, 'test_label.npy'), self.Y)


if __name__ == '__main__':
    dataset = 'public'  # 'dataset1'
    path = "D:/YS/M_D/20200125/Test"
    dataset = Dataset(path, dataset)

    print(f"split train/valid/test dataset ...")
    dataset.test_data()

