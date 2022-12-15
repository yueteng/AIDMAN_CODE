#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2020/12/10 15:59
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
from utils.attention import AttentionLayer
from model.transformer4 import LayerNormalization, TokenAndPositionEmbedding
from keras_transformer.attention import MultiHeadSelfAttention, MultiHeadAttention
# from model.UNet import unet
# from model.SEGNet import segnet
# from model.PSPNet import pspnet
# from model.AttRep import att_reps
from model.AIM import AIM
# from model.multi_scale_vgg import multi_scale_vgg, aggregate_vgg, aggregate_norm_vgg
from model.multi_scale_att import multi_scale_att, multi_scale_han
# from model.multi_scale_att_0 import multi_scale_transformer, AIM_without_multi_scale_attention, AIM_without_local_context_aligner
from model.multi_scale_trans_scale_n import multi_scale_transformer_3,multi_scale_transformer_5_head_4,multi_scale_transformer_5_head_8,multi_scale_transformer_4_head16,multi_scale_transformer_4_head4,multi_scale_transformer_5,multi_scale_transformer_1,multi_scale_transformer_2
from model.encoding_transformer import multi_scale_encoding_transformer
# from model.cluster import cluster_claffication
from utils import module
from utils.attention import AttentionLayer
from utils.classification import roc_list, fold_roc_list,all_fold_roc_list
import tensorflow as tf
import keras
from database1 import Dataset
from model.SENet import SEResNeXt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tf_config)
def set_parameters():
    parser = argparse.ArgumentParser(description="Malaria Detection")
    parser.add_argument('--dataset', type=str, default='data/dataset1')
    parser.add_argument('--train_mode', type=str, default='train', )
    parser.add_argument('--model_save_path', type=str, default='./output/')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--seed', type=int, default=2020)

    parser.add_argument('--model', type=str, default='multi_scale_transformer_4')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch_num', type=int, default=1000)
    parser.add_argument('--stop_num', type=int, default=50)
    parser.add_argument('--optimer', type=str, default="adam")


    parser.add_argument('--pretrain', type=bool, default=False)

    parser.add_argument('--fold', type=int, default=0)

    config = parser.parse_args()
    return config

def load_cross_data(config):
    """ loading the sample data """
    # if config.train_mode == 'train':
    path = f'./{config.dataset}/train_valid_test'
    data_0 = np.load(os.path.join(path, "test_data.npy")).tolist()
    lable_0 = np.load(os.path.join(path, "test_label.npy")).tolist()

    data_1 = np.load(os.path.join(path, "test_data.npy")).tolist()
    lable_1 = np.load(os.path.join(path, "test_label.npy")).tolist()

    data_2 = np.load(os.path.join(path, "test_data.npy")).tolist()
    lable_2 = np.load(os.path.join(path, "test_label.npy")).tolist()

    data_3 = np.load(os.path.join(path, "test_data.npy")).tolist()
    lable_3 = np.load(os.path.join(path, "test_label.npy")).tolist()

    data_4 = np.load(os.path.join(path, "test_data.npy")).tolist()
    lable_4 = np.load(os.path.join(path, "test_label.npy")).tolist()

    if config.fold == 0:
        train_X = np.array(data_0 + data_1 + data_2)
        train_Y = np.array(lable_0 + lable_1 + lable_2)

        valid_X = np.array(data_3)
        valid_Y = np.array(lable_3)

        test_X = np.array(data_4)
        test_Y = np.array(lable_4)
    elif config.fold == 1:
        train_X = np.array(data_1 + data_2 + data_3)
        train_Y = np.array(lable_1 + lable_2 + lable_3)

        valid_X = np.array(data_4)
        valid_Y = np.array(lable_4)

        test_X = np.array(data_0)
        test_Y = np.array(lable_0)
    elif config.fold == 2:
        train_X = np.array(data_2 + data_3 + data_4)
        train_Y = np.array(lable_2 + lable_3 + lable_4)

        valid_X = np.array(data_0)
        valid_Y = np.array(lable_0)

        test_X = np.array(data_1)
        test_Y = np.array(lable_1)
    elif config.fold == 3:
        train_X = np.array(data_3 + data_4 + data_0)
        train_Y = np.array(lable_3 + lable_4 + lable_0)

        valid_X = np.array(data_1)
        valid_Y = np.array(lable_1)

        test_X = np.array(data_2)
        test_Y = np.array(lable_2)
    elif config.fold == 4:
        train_X = np.array(data_4 + data_0 + data_1)
        train_Y = np.array(lable_4 + lable_0 + lable_1)

        valid_X = np.array(data_2)
        valid_Y = np.array(lable_2)

        test_X = np.array(data_3)
        test_Y = np.array(lable_3)

    datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, rotation_range=20,
                                 horizontal_flip=True, vertical_flip=True, brightness_range=[0.5, 1])

    train_generator = datagen.flow(train_X, train_Y, batch_size=config.batch_size)
    config.train_num = train_X.shape[0]


    datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = datagen.flow(valid_X, valid_Y, batch_size=1)
    config.valid_num = valid_X.shape[0]

    test_generator = datagen.flow(test_X, test_Y, batch_size=1)
    config.test_num = test_X.shape[0]

    return {"train": train_generator, "valid": valid_generator, "test": test_generator,
            "valid_X": valid_X, "valid_Y": valid_Y, "test_X": test_X, "test_Y": test_Y, }



def load_data(config):
    """ loading the sample data """
    # if config.train_mode == 'train':
    path = f'./{config.dataset}/train_valid_test'
    print(path)
    train_X = np.load(os.path.join(path, "train_data.npy"))
    train_Y = np.load(os.path.join(path, "train_label.npy"))

    valid_X = np.load(os.path.join(path, "valid_data.npy"))
    valid_Y = np.load(os.path.join(path, "valid_label.npy"))

    test_X = np.load(os.path.join(path, "test_data.npy"))
    test_Y = np.load(os.path.join(path, "test_label.npy"))

    datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, rotation_range=20,
                                 horizontal_flip=True, vertical_flip=True, brightness_range=[0.5, 1])

    train_generator = datagen.flow(train_X, train_Y, batch_size=config.batch_size)
    config.train_num = train_X.shape[0]


    datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = datagen.flow(valid_X, valid_Y, batch_size=1)
    config.valid_num = valid_X.shape[0]

    test_generator = datagen.flow(test_X, test_Y, batch_size=1)
    config.test_num = test_X.shape[0]

    return {"train": train_generator, "valid": valid_generator, "test": test_generator,
            "valid_X": valid_X, "valid_Y": valid_Y, "test_X": test_X, "test_Y": test_Y, }


def load_model(config):
    models = {
        # 'unet': unet,
        # 'segnet': segnet,
        # 'pspnet': pspnet,
        #'ResNet': module.res50(),
        #'VGG': module.vgg(),
        # 'att_rep': att_reps,
        #'aim': AIM,

        # 'multi_scale_vgg': multi_scale_vgg,
        # 'agg_vgg': aggregate_vgg,
        # 'agg_norm_vgg': aggregate_norm_vgg,

        # 'multi_scale_att': multi_scale_att,
        # 'multi_scale_han': multi_scale_han,
        # 'AIM_without_multi_scale_attention': AIM_without_multi_scale_attention(),
        # 'AIM_without_local_context_aligner': AIM_without_local_context_aligner(),
        #'multi_scale_transformer': multi_scale_transformer(),
        # "multi_scale_encoding_transformer":multi_scale_encoding_transformer(),
        'multi_scale_transformer_3': multi_scale_transformer_3(),
        'multi_scale_transformer_4_head4': multi_scale_transformer_4_head4(),
        'multi_scale_transformer_5_head_8': multi_scale_transformer_5_head_8(),
        'multi_scale_transformer_5_head_4': multi_scale_transformer_5_head_4(),

        'multi_scale_transformer_4_head16': multi_scale_transformer_4_head16(),
        'multi_scale_transformer_5': multi_scale_transformer_5(),
        'multi_scale_transformer_2': multi_scale_transformer_2(),
        'multi_scale_transformer_1': multi_scale_transformer_1(),
        # 'multi_scale_transformer_0': multi_scale_transformer_0(),
        #'SENet': SEResNeXt(64, 1).model
    }
    assert config.model in models

    return models[config.model]


def train(config):
    """ Training Model """


    dataset = load_data(config)
    # if config.pretrain:
    #     model_save_path = f"{config.model_save_path}/public_{config.train_mode}_{config.model}_{str(config.fold)}.h5"
    #     model = keras_load_model(model_save_path, custom_objects={'tf': tf,
    #                                                           'AttentionLayer': AttentionLayer})
    # else:
    model = load_model(config)
    print(model.summary())

    model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}_{str(config.fold)}.h5"
    print("model save path")
    print(model_save_path)
    # early_stopping = EarlyStopping(monitor='val_acc', patience=config.stop_num, verbose=0, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc', patience=config.stop_num, verbose=0, mode='max')
    check_point = ModelCheckpoint(model_save_path, monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    callbacks_list = [check_point, early_stopping]

    #training the model
    history = model.fit_generator(dataset['train'],
                                  steps_per_epoch=config.train_num/config.batch_size,
                                  epochs=config.max_epoch_num,
                                  validation_data=dataset['valid'],
                                  validation_steps=config.valid_num,
                                  callbacks=callbacks_list, verbose=2)
    return model
    #Plotting Training and Testing accuracies


def evaluate(config):
    dataset = load_cross_data(config)
    valid_X = dataset["valid_X"]
    valid_Y = dataset["valid_Y"]

    model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}_{str(config.fold)}.h5"
    model = keras_load_model(model_save_path, custom_objects={'tf': tf, 'stack': K.stack,
                                                              'AttentionLayer': AttentionLayer})


    # obtaining accuracy on test set
    val_acc = model.evaluate(valid_X/255.0, valid_Y, verbose=0)

    print(model.metrics_names)
    print('Valid Accuracy Obtained: ')
    print(val_acc[1] * 100, ' %')

    test_X = dataset["test_X"]
    test_Y = dataset["test_Y"]
    test_acc = model.evaluate(test_X / 255.0, test_Y, verbose=2)
    print('Test Accuracy Obtained: ')
    print(test_acc[1] * 100, ' %')

def allroc(config):

    # model_list = [ "multi_scale_transformer_2.h5", "multi_scale_att_1.h5", "multi_scale_transformer_949.h5"]

    p_list = []
    y_list = []

    for fold in range(1):
        fold=0
        config.fold = fold
        dataset = load_cross_data(config)
        x = dataset["test_X"]
        y = dataset["test_Y"]

        model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}_{fold}" + ".h5"
        print(model_save_path)
        config.model=model_save_path
        model = keras_load_model(model_save_path, custom_objects={'tf': tf, 'stack': K.stack,
                                                                  'AttentionLayer': AttentionLayer,
                                                                  'MultiHeadAttention':MultiHeadAttention,
                                                                  'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                                                  'LayerNormalization':LayerNormalization,
                                                                  'MultiHeadSelfAttention':MultiHeadSelfAttention
                                                                  })
        model.summary()
        test_acc = model.evaluate(x / 255.0, y, verbose=2, )
        print('Test Accuracy Obtained: ' + str(fold))
        print(test_acc[1])
        p = model.predict(x/255.0)

        num = 0
        for i in range(len(p)):
            if np.round(p[i]) != y[i]:
                num += 1
                #print(pic_name[i])
        print("ACC: " + str(1 - num/len(p)))
        p = np.squeeze(p)
        y_list.append(y)

        p_list.append(p)
    roc_list(y_list, p_list, config.model)






def predict(x, y, pic_name):


    model_save_path = f"{config.model_save_path}/{config.dataset}_{config.train_mode}_{config.model}_{str(config.fold)}.h5"

    model = keras_load_model(model_save_path, custom_objects={'tf': tf, 'stack': K.stack,
                                                              'AttentionLayer': AttentionLayer
                                                              ,'TokenAndPositionEmbedding':TokenAndPositionEmbedding})
    print(model.summary())
    p = model.predict(x/255.0)
    num = 0
    for i in range(len(p)):
        if np.round(p[i]) != y[i]:
            num += 1
            #print(pic_name[i])
    print(1 - num/len(p))

    keras.backend.clear_session()

if __name__ == "__main__":
    config = set_parameters()


    print(config)
    # model=train(config)
    # evaluate(config)
    # path = "D:/YS/M_D/20210125/Test"
    # x, y, pic_name = Dataset(path).prediction_data()
    # # predict(x, y, pic_name)
    allroc(config)
    # keras.backend.clear_session()