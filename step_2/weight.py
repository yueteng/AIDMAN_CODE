#!usr/bin/env python
#coding:utf-8

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
from keras.models import Model
from model.UNet import unet
from model.SEGNet import segnet
from model.PSPNet import pspnet
from model.AttRep import att_reps
from model.AIM import AIM
from model.multi_scale_vgg import multi_scale_vgg, aggregate_vgg, aggregate_norm_vgg
from model.multi_scale_att import multi_scale_att, multi_scale_han, reshapes
from model.cluster import cluster_claffication
from utils import module
from utils.attention import AttentionLayer
from model.transformer4 import TokenAndPositionEmbedding
import tensorflow as tf
import keras
from database1 import Dataset

from keras.layers import *      #For adding convolutional layer
from keras.layers import Dense, ZeroPadding2D, BatchNormalization, Activation, concatenate        #For adding layers to NN
from keras.optimizers import adam
from keras.models import *                  #for loading the model
from keras.backend import stack, squeeze, expand_dims
from utils import module
from keras import backend as K
from utils.attention import AttentionLayer
import glob
from PIL import Image

def load_images( images_list):
    data = list()
    for image_path in images_list:
        img = Image.open(image_path)
        resized_img = img.resize((64, 64))
        img_array = np.array(resized_img)
        data.append(img_array)
    return data


def set_parameters():
    parser = argparse.ArgumentParser(description="Malaria Detection")
    parser.add_argument('--dataset', type=str, default='dataset3')
    parser.add_argument('--train_mode', type=str, default='train', )
    parser.add_argument('--model_save_path', type=str, default='./output/')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--model', type=str, default='multi_scale_transformer_2')
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
    valid_generator = datagen.flow(valid_X, valid_Y, batch_size=config.batch_size)
    config.valid_num = valid_X.shape[0]

    test_X = np.load(os.path.join(path, "test_data.npy"))
    test_Y = np.load(os.path.join(path, "test_label.npy"))
    test_generator = datagen.flow(test_X, test_Y, batch_size=config.batch_size)
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
        'multi_scale_han': multi_scale_han,
        'cluster_classification': cluster_claffication,
    }
    assert config.model in models
    return models[config.model]

def weight(config):
    img_path = os.path.join(path, 'attention')
    images = glob.glob(img_path + '/*.png')
    print(images)
    X = list()
    X.extend(load_images(images))
    X = np.array(X)[0:1000]

    model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}.h5"
    print(model_save_path)
    model = keras_load_model(model_save_path, custom_objects={'tf': tf, 'stack': K.stack,
                                                              'AttentionLayer': AttentionLayer,
                                                              'reshapes':reshapes,
                                                              "TokenAndPositionEmbedding": TokenAndPositionEmbedding})

    print(model.summary())

    lambda2_model = Model(inputs=model.input, outputs=model.get_layer("reshape_6").output)
    lambda2_out = lambda2_model.predict(X/ 255.0)

    W = model.get_layer("attention")._get_attention_weights(lambda2_out)
    W = K.eval(W)
    for i in range(len(W)):

            print(W[i])
            print(images[i])
    p = model.predict(X/255.0)
    p = np.round(p)
    print(np.round(p))

def predict():
    img_path = os.path.join(path, 'attention')
    images = glob.glob(img_path + '/*.png')
    print(images)
    X = list()
    X.extend(load_images(images))
    X = np.array(X)[0:1000]

    model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}.h5"

    print(model_save_path)
    model = keras_load_model(model_save_path, custom_objects={'tf': tf, 'stack': K.stack,
                                                              'AttentionLayer': AttentionLayer, 'reshapes': reshapes})
    print(model.summary())
    p = model.predict(X/255.0)
    p = np.round(p)
    p = np.squeeze(p).tolist()
    for i in range(len(p)):
        if p[i] == 1.0:
            print(images[i])



if __name__ == "__main__":

    config = set_parameters()
    print(config)

    path = r"D:\YS\M_D\20210118"
    #weight(config)
    path = r"D:\YS\M_D\20210118"
    predict()

    keras.backend.clear_session()