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
import cv2
from model.transformer4 import TokenAndPositionEmbedding
from utils.attention import AttentionLayer
import tensorflow as tf
import glob
from PIL import Image
import shutil
import csv

def set_parameters():
    parser = argparse.ArgumentParser(description="Malaria Detection")
    parser.add_argument('--image', type=str, default='train_data/Positivate_sp')
    parser.add_argument('--dataset', type=str, default='head')
    parser.add_argument('--model_save_path', type=str, default='output1/')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--positive', type=str, default='positive')
    parser.add_argument('--model', type=str, default='multi_scale_transformer')
    parser.add_argument("--train_mode", type=str, default="train")
    parser.add_argument("--flag", type=str, default="4")
    config = parser.parse_args()
    return config

def load_images( images_list):
    data = list()
    for image_path in images_list:
        img = Image.open(image_path)
        resized_img = img.resize((64, 64))
        img_array = np.array(resized_img)
        data.append(img_array)
    return data

def check_file(filename):
    # if os.path.exists(filename):
    #     shutil.rmtree(filename)

    path = filename

    if os.path.exists(path):
        print('exist')
    else:
        os.mkdir(path)


def load_data(config):

    dataset = config.image

    imagespath = glob.glob(dataset + '/*.jpg')
    imagespath += glob.glob(dataset + '/*.png')
    X = list()
    X.extend(load_images(imagespath))
    return X, imagespath


def prediction(config, csv_data):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda

    dataset, imagespath = load_data(config)
    dataset = np.array(dataset)
    model_save_path = "best_model_step_2.h5"
    model = keras_load_model(model_save_path, custom_objects={ 'stack': K.stack,
                                                              'AttentionLayer': AttentionLayer,
                                                               'TokenAndPositionEmbedding': TokenAndPositionEmbedding})

    # obtaining accuracy on test set
    predict = model.predict(dataset/255.0)
    # predict = np.array(np.round(predict), np.int32)
    # x_index = np.where(predict)
    # for i in x_index[0]:
    #     cv2.imwrite(config.positive +"/"+ str(i)+".png", dataset[i])
    # print(sum(predict))
    predict = np.squeeze(predict)
    x_index = np.argsort(predict)[::-1]
    flag = 0
    aaa=config.positive + "/" + "_".join(imagespath[x_index[0]].split("\\")[-1].split("_")[:-1]) + ".csv"

    with open(config.positive + "/" + "_".join(imagespath[x_index[0]].split("\\")[-1].split("_")[:-1]) + ".csv",
              "w") as f:
        f.write("no,id,pred\n")
        for i in x_index[0:64]:
            f.write(str(flag)+","+str(i)+","+str(predict[i])+"\n")
            cv2.imwrite(config.positive + "/" + "_".join(imagespath[x_index[i]].split("\\")[-1].split("_")[:-1])+"_"+str(flag)+".jpg", dataset[i])


            # # img = cv2.imread(config.positive +"/"+ imagespath[x_index[i]].split("\\")[-1])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(config.positive + "/" + "_".join(imagespath[x_index[i]].split("\\")[-1].split("_")[:-1])+"_"+str(i)+"jpg", img)
            # # cv2.imwrite(config.positive +"/"+ imagespath[x_index[i]].split("\\")[-1], dataset[i])
            flag += 1


            csv_data.append([config.positive.split("/")[-1] , str(flag)])

    K.clear_session()

if __name__ == "__main__":
    #Positivate
    classimg = "Positivate"
    config = set_parameters()
    print(config)
    csv_data = [["name", "num"]]
    path = "train_data_20220824/train_data/" + classimg + "_sp"  #检测图片文件夹
    from tqdm import tqdm
    aaa=os.listdir(path)
    for file in tqdm(os.listdir(path)):
    # for file in ["20200103_S185"]:
        check_file("train_data_20220824/step_2/" + classimg + "/" + file) #预测结果存放位置
        config.image = path+ "/" + file
        config.positive = "train_data_20220824/step_2/" + classimg + "/"+ file#预测结果存放位置
        prediction(config, csv_data)
    import pyexcel as pe

    sheet = pe.Sheet(csv_data)
    sheet.save_as(config.flag + "train_data_20220824/train_data/" + classimg + ".csv")


# if __name__ == "__main__":
#     config = set_parameters()
#     print(config)
#     csv_data = [["name", "num"]]
#     path = "Train_select/Parasitized"  #检测图片文件夹
#
#     check_file("train_prediction/Parasitized/") #预测结果存放位置
#     config.image = path+ "/"
#     config.positive = "train_prediction/Parasitized/"#预测结果存放位置
#     prediction(config, csv_data)
#     import pyexcel as pe
#
#     sheet = pe.Sheet(csv_data)
#     sheet.save_as("train_prediction_Parasitized.csv")
#
