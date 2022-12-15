#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2021/1/10 8:58
@project: MalariaDetection
@description: 
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model as keras_load_model
from keras import backend as K

from model.UNet import unet
from model.SEGNet import segnet
from model.PSPNet import pspnet
from model.AttRep import att_reps
from model.AIM import AIM
from model.multi_scale_vgg import multi_scale_vgg, aggregate_vgg, aggregate_norm_vgg
from model.multi_scale_att import multi_scale_att
from utils import module
from utils.attention import AttentionLayer
import tensorflow as tf


def set_parameters():
    parser = argparse.ArgumentParser(description="Malaria Detection")
    parser.add_argument('--dataset', type=str, default='dataset1')
    parser.add_argument('--train_mode', type=str, default='train', )
    parser.add_argument('--model_save_path', type=str, default='./output/')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--model', type=str, default='multi_scale_att')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch_num', type=int, default=1000)
    parser.add_argument('--stop_num', type=int, default=20)
    parser.add_argument('--optimer', type=str, default="adam")

    parser.add_argument('--pretrain', type=bool, default=False)

    config = parser.parse_args()
    return config


def load_data(config):
    """ loading the sample data """
    # if config.train_mode == 'train':
    path = f'./data/{config.dataset}/train_valid_test'
    train_X = np.load(os.path.join(path, "train_data.npy"))
    train_Y = np.load(os.path.join(path, "train_label.npy"))

    datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, rotation_range=20,
                                 horizontal_flip=True, vertical_flip=True, brightness_range=[0.5, 1])
    train_generator = datagen.flow(train_X, train_Y, batch_size=config.batch_size)
    config.train_num = train_X.shape[0]

    valid_X = np.load(os.path.join(path, "valid_data.npy"))
    valid_Y = np.load(os.path.join(path, "valid_label.npy"))
    datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = datagen.flow(valid_X, valid_Y, batch_size=config.batch_size, shuffle=False)
    config.valid_num = valid_X.shape[0]

    test_X = np.load(os.path.join(path, "test_data.npy"))
    test_Y = np.load(os.path.join(path, "test_label.npy"))
    test_generator = datagen.flow(test_X, test_Y, batch_size=config.batch_size, shuffle=False)
    config.test_num = test_X.shape[0]

    return {"train": train_generator, "valid": valid_generator, "test": test_generator}


def load_model(config):
    models = {
        'unet': unet,
        'segnet': segnet,
        'pspnet': pspnet,
        'resnet': module.res50,
        'vgg': module.vgg,
        'att_rep': att_reps,
        'aim': AIM,

        'multi_scale_vgg': multi_scale_vgg,
        'agg_vgg': aggregate_vgg,
        'agg_norm_vgg': aggregate_norm_vgg,

        'multi_scale_att': multi_scale_att,
    }
    assert config.model in models
    return models[config.model]


def evaluate(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda

    dataset = load_data(config)

    model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}.h5"
    model = keras_load_model(model_save_path, custom_objects={'tf': tf, 'stack': K.stack,
                                                              'AttentionLayer': AttentionLayer})

    # obtaining accuracy on test set
    test_acc = model.evaluate_generator(dataset['valid'], steps=config.valid_num)
    print(model.metrics_names)
    print('Valid Accuracy Obtained: ')
    print(test_acc[1] * 100, ' %')

    # obtaining accuracy on test set
    predict = model.predict_generator(dataset['valid'], verbose=1)
    predict = np.array(np.round(predict), np.int32)
    predict = np.reshape(predict, [-1])
    acc_num = np.sum(predict == dataset['valid'].y)

    print('Valid Accuracy Obtained: ', float(acc_num)/config.valid_num)

    # obtaining accuracy on test set
    test_acc = model.evaluate_generator(dataset['test'], steps=config.test_num)
    print(model.metrics_names)
    print(model.metrics_names)
    print('Test Accuracy Obtained: ')
    print(test_acc[1] * 100, ' %')

    # obtaining accuracy on test set
    predict = model.predict_generator(dataset['test'], verbose=1)
    predict = np.array(np.round(predict), np.int32)
    predict = np.reshape(predict, [-1])
    acc_num = np.sum(predict == dataset['test'].y)

    print('Test Accuracy Obtained: ', float(acc_num)/config.test_num)

    # obtaining accuracy on test set
    predict = model.predict(dataset['test'].x * 1.0/255)
    predict = np.array(np.round(predict), np.int32)
    predict = np.reshape(predict, [-1])
    acc_num = np.sum(predict == dataset['test'].y)

    print('Test Accuracy Obtained: ', float(acc_num) / config.test_num)


if __name__ == "__main__":
    config = set_parameters()
    print(config)
    evaluate(config)
