# coding:utf-8
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from keras.optimizers import adam
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Model, load_model
import glob
from PIL import Image
import math
import random
import cv2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pyexcel as pe
import keras
from imblearn.metrics import classification_report_imbalanced
import json
from sklearn.metrics import roc_curve, auc, precision_score, recall_score,f1_score
import matplotlib.pyplot as plt




def load_images( images_list):
	img = np.array(Image.open(images_list))
	return img


def load_data(config):
	dataset = config.image_Pos
	X, Y = list(), list()

	for file in glob.glob(dataset + '/*.jpg'):
		X.append(load_images(file))
		Y.append(1)

	dataset = config.image_Neg
	for file in glob.glob(dataset + '/*.jpg'):
		X.append(load_images(file))
		Y.append(0)
	return X, Y


from sklearn.utils import shuffle


def load_splited_imgs(config):
	data={}
	for data_set in ["train","val","test"]:
		X=[]
		Y=[]
		for file in glob.glob(config.image_Pos+"/"+data_set + '/*.jpg'):
			X.append(load_images(file))
			Y.append(1)
		for file in glob.glob(config.image_Neg+"/"+data_set + '/*.jpg'):
			X.append(load_images(file))
			Y.append(0)
		X, Y = shuffle(X, Y, random_state=0)
		if data_set=="train":
			data["Train_X"]=np.array(X)
			data["Train_Y"] = np.array(Y)
		elif data_set=="val":
			data["Valid_X"]=np.array(X)
			data["Valid_Y"] = np.array(Y)
		elif data_set=="test":
			data["Test_X"]=np.array(X)
			data["Test_Y"] = np.array(Y)
		print()

	return data

def split_data(config):
	
	X, Y = load_data(config)
	index = np.arange(len(Y))
	np.random.shuffle(index)
	
	tran_len = int(np.round(len(Y)*config.train_percentage))
	valid_len = int(tran_len + np.round((len(Y)-tran_len)/2))
	train_index = index[:tran_len]
	valid_index = index[tran_len:valid_len]
	test_index = index[valid_len:]
	X, Y = np.array(X), np.array(Y)
	Train_X = X[train_index]
	Train_Y = Y[train_index]
	
	Valid_X = X[valid_index]
	Valid_Y = Y[valid_index]
	
	Test_X = X[test_index]
	Test_Y = Y[test_index]

	data={"Train_X": Train_X, "Train_Y": Train_Y,
	 "Valid_X": Valid_X, "Valid_Y": Valid_Y,
	 "Test_X": Test_X, "Test_Y": Test_Y}
	np.save(model_save_path+".npy",np.array([data]))
	return data

def fc_model():
	input = Input(shape=(int(math.sqrt(num_concat))*64,int(math.sqrt(num_concat))*64, 3))

	x = Conv2D(128, (3, 3), activation="relu")(input)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2))(x)

	x = Conv2D(64, (3, 3), activation="relu")(x)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2))(x)

	x = Conv2D(32, (3, 3), activation="relu")(x)
	x = BatchNormalization()(x)
	x = MaxPool2D((2, 2))(x)


	x = Flatten()(x)
	x = Dense(64, activation="relu")(x)
	x = BatchNormalization()(x)
	# x = Dense(64, activation="relu")(x)
	y = Dense(1, activation="sigmoid")(x)


	
	model = Model(input= input, output= y)
	model.compile(optimizer=adam(lr=0.00001), loss='binary_crossentropy',
				  metrics=['accuracy'])

	return model

def set_parameters():
	parser = argparse.ArgumentParser(description="step_3")
	parser.add_argument('--image_Neg', type=str, default=path+"Negtivate/")
	parser.add_argument('--image_Pos', type=str, default=path+"Positivate/")
	parser.add_argument('--cuda', type=str, default='0')
	parser.add_argument("--train_percentage", type=float, default=0.6)

	config = parser.parse_args()

	return config




def roc_lists(y_test, test_predict, model):
	y_test_int = np.round(test_predict).astype("int32")
	report = classification_report_imbalanced(y_test, y_test_int, digits=5)
	print(report)
	fpr, tpr, threshold = roc_curve(y_test, test_predict)
	roc_auc = auc(fpr, tpr)
	print("AUC:", roc_auc)
	lw = 2
	plt.figure(figsize=(10, 10))
	plt.plot(fpr, tpr, color="red",
			 lw=lw, label='AUC = %0.5f' % roc_auc)
	# plt.legend(loc="4")

	###假正率为横坐标，真正率为纵坐标做曲线
	plt.rcParams["font.weight"] = "bold"
	plt.rcParams["axes.labelweight"] = "bold"
	plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.xlabel('False Positive Rate', fontsize=20)
	plt.ylabel('True Positive Rate', fontsize=20)
	plt.title('CNN', fontsize=20)
	# plt.title(model)
	plt.legend(loc="lower right", fontsize=20)
	plt.grid()
	print(model + "_ROC.png")
	plt.savefig(model + "_ROC.png", dpi=600)


import os, random, shutil



if __name__ == "__main__":
	num_concat=25
	path="train_data_20220824/Image_concat_ori/thres_"+str(num_concat)+"/"
	model_save_path = path+"/v2_best_model_"+str(num_concat)+".h5"
	config = set_parameters()
	# moveFile()
	model = fc_model()
	model.summary()
	print(config)
	# if os.path.exists(model_save_path+".npy")==False:
		# data = split_data(config)
		# data=np.load("train_data_20220824/Image_concat/train_data_step_3.npy",allow_pickle=True)[0]

	data=load_splited_imgs(config)
	early_stopping = EarlyStopping(monitor='val_acc', patience=100, verbose=0, mode='max')
	check_point = ModelCheckpoint(model_save_path, monitor='val_acc', mode='max', verbose=2, save_best_only=True)
	callbacks_list = [check_point, early_stopping]



	model.fit(x=data['Train_X'], y=data["Train_Y"], epochs=100, batch_size=6,validation_data=(data['Valid_X'], data["Valid_Y"]),callbacks=callbacks_list, verbose=2)

	# else:
	# 	data=np.load(model_save_path + ".npy",allow_pickle=True)[0]



	model=load_model(model_save_path)
	print(model.evaluate(data["Test_X"],data["Test_Y"]))
	print(model.evaluate(data["Test_X"],data["Test_Y"]))
	pre = model.predict(x=data['Test_X'])
	pre = np.squeeze(pre)
	roc_lists(data["Test_Y"], pre, model_save_path)
	keras.backend.clear_session()
	# model=load_model("SmartMalariaNET-master/SmartMalariaNET-master/AIDMAN/The Third dataset/best_model.h5")
	# model.summary()
	# data=np.load("SmartMalariaNET-master/SmartMalariaNET-master/AIDMAN/The Third dataset/test_data.npy")
	# label=np.load("SmartMalariaNET-master/SmartMalariaNET-master/AIDMAN/The Third dataset/test_label.npy")
	# print(model.evaluate(data,label))
