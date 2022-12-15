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
from model.transformer4 import TokenAndPositionEmbedding
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
from sklearn.metrics import auc, recall_score, precision_score, accuracy_score
from utils.classification import roc_list, fold_roc_list,all_fold_roc_list

def set_parameters():
    parser = argparse.ArgumentParser(description="Malaria Detection")
    parser.add_argument('--dataset', type=str, default='dataset1')
    parser.add_argument('--train_mode', type=str, default='train', )
    parser.add_argument('--model_save_path', type=str, default='./output1')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--model', type=str, default='multi_scale_transformer')
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
    path = f'./data/{config.dataset}/cross_fold'

    datagen = ImageDataGenerator(rescale=1. / 255)

    ind_X = np.load(os.path.join(path, "data_independent.npy"))
    ind_Y = np.load(os.path.join(path, "label_independent.npy"))
    ind_generator = datagen.flow(ind_X, ind_Y, batch_size=config.batch_size, shuffle=False)
    config.test_num = ind_X.shape[0]



    return {"ind": ind_generator, "ind_x":ind_X, "ind_y":ind_Y}

def allroc(config):

    # model_list = [ "multi_scale_transformer_2.h5", "multi_scale_att_1.h5", "multi_scale_transformer_949.h5"]

    p_list = []
    y_list = []

    for fold in range(5):
        config.fold = fold
        dataset = load_data(config)
        x = dataset["ind_x"]
        y = dataset["ind_y"]

        model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}_{fold}" + ".h5"
        model = keras_load_model(model_save_path, custom_objects={'tf': tf, 'stack': K.stack,
                                                                  'AttentionLayer': AttentionLayer,
                                                                  "TokenAndPositionEmbedding":TokenAndPositionEmbedding})

        p = model.predict(x/255.0)

        num = 0
        for i in range(len(p)):
            if np.round(p[i]) != y[i]:
                num += 1
                #print(pic_name[i])
        print("ACC: ", accuracy_score(y, np.round(p)))
        print("RECALL:", recall_score(y, np.round(p)))
        print("PRECISION:", precision_score(y, np.round(p)))
        print("AUC:", auc(y, np.squeeze(p)))
        p = np.squeeze(p)
        y_list.append(y)
        p_list.append(p)
    #roc_list(y_list, p_list, config.model)



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


    ind_acc = model.evaluate_generator(dataset['ind'], steps=config.test_num)
    print(model.metrics_names)
    print('Ind Accuracy Obtained: ')
    print(ind_acc[1] * 100, ' %')

    # obtaining accuracy on test set
    predict = model.predict_generator(dataset['ind'], verbose=1)
    predict = np.array(np.round(predict), np.int32)
    predict = np.reshape(predict, [-1])
    acc_num = np.sum(predict == dataset['ind'].y)

    print('Ind Accuracy Obtained: ', float(acc_num)/config.test_num)

    # obtaining accuracy on test set
    predict = model.predict(dataset['ind'].x * 1.0/255)
    predict = np.array(np.round(predict), np.int32)
    predict = np.reshape(predict, [-1])
    acc_num = np.sum(predict == dataset['ind'].y)

    print('Ind Accuracy Obtained: ', float(acc_num) / config.ind_num)


if __name__ == "__main__":
    config = set_parameters()
    print(config)
    allroc(config)
