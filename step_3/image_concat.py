# coding:utf-8
import argparse
import os
import numpy as np
import glob
from PIL import Image
import cv2
import math
import random
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pyexcel as pe
import pandas as pd
import shutil
def check_file(filename):
    if os.path.exists(filename):
        shutil.rmtree(filename)

    path = filename

    if os.path.exists(path):
        print('exist')
    else:
        os.mkdir(path)


# def load_images(images_list):
#
#     data = [0]*5
#
#     for i in range(5):
#         flag = 0
#         for image_path in images_list[i*5:(i+1)*5]:
#             img = Image.open(image_path)
#             resized_img = img.resize((64, 64))
#             img_array = np.array(resized_img)
#
#             if flag == 0:
#                 data[i] = img_array
#                 flag += 1
#             else:
#
#                 data[i] = np.hstack((data[i], img_array))
#                 flag += 1
#     img = np.vstack((data[0], data[1], data[2], data[3], data[4]))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     cv2.imwrite("Image_concat/RDT/" + image_path.split("/")[-1].split("\\")[0] + ".jpg", img)
#

def load_images(dataset, images_list):


    data = [0]*int(math.sqrt(thres))

    for i in range(int(math.sqrt(thres))):
        flag = 0
        for image_path in images_list[i*int(math.sqrt(thres)):(i+1)*int(math.sqrt(thres))]:
            try:
                img = Image.open(image_path)
                img_temp=img.copy()
            except:
                img = img_temp
            resized_img = img.resize((64, 64))
            img_array = np.array(resized_img)

            if flag == 0:
                data[i] = img_array
                flag += 1
            else:

                data[i] = np.hstack((data[i], img_array))
                flag += 1
    stake=[]
    for i in range(int(math.sqrt(thres))):
        stake.append(data[i])
    img = np.vstack(stake)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("save to ===================== "+dataset + file+".jpg")
    cv2.imwrite(dataset + file+".jpg", img)






def img_concat(dataset, file):

    images=[]
    for i in range(thres):
        images.append(img_path+ file +"/"+file+"_cell_"+str(i)+".jpg")

    image_save_path="train_data_20220824/Image_concat/"+"/thres_"+str(thres)+"/"
    if os.path.exists(image_save_path)==False:
        os.mkdir(image_save_path)

    image_save_path += img_path.split("/")[-2]+"/"

    if os.path.exists(image_save_path)==False:
        os.mkdir(image_save_path)

    load_images(image_save_path, images)

if __name__ == "__main__":
    for thres in range(8):
        thres+=1
        thres=thres*thres
        for folder in ["Positivate","Negtivate"]:
            img_path="train_data_20220824/CAM/"+folder+"/"
            for file in os.listdir(img_path):
                img_concat("CAM", file)



