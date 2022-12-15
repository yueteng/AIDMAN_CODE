# -*- coding=utf-8 -*-
"""
Created on 2019-6-19 21:39:53
@author: fangsh.Alex
"""
import os
from attention import AttentionLayer

from model.transformer4 import TokenAndPositionEmbedding
num = "1"
import keras
import cv2
import numpy as np
import keras.backend as K
from keras import initializers
from PIL import Image
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import glob
import tensorflow as tf
from keras import layers
import shutil
import sys

class AttentionLayer(keras.layers.Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def _get_attention_weights(self, X):
        uit = K.tanh(K.bias_add(K.dot(X, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        return ait

    def get_config(self):
        config = {
            'attention_dim': self.attention_dim
        }
        base_config = super(AttentionLayer, self).get_config()
        return {**base_config, **config}


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_rows, max_cols, embed_dim,  **kwargs):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.embed_dim = embed_dim
        super(TokenAndPositionEmbedding, self).__init__()

    def build(self, input_shape):
        row_pos_emb = np.zeros((self.max_rows, self.embed_dim), dtype=np.float)
        col_pos_emb = np.zeros((self.max_cols, self.embed_dim), dtype=np.float)
        for i in range(self.max_rows):
            tmp = np.arange(self.embed_dim)
            row_pos_emb[i, ::2] = np.sin(i/10000.0**(tmp[::2]/float(self.embed_dim)))
            row_pos_emb[i, 1::2] = np.cos(i/10000.0**(tmp[1::2]/float(self.embed_dim)))

        for i in range(self.max_cols):
            tmp = np.arange(self.embed_dim)
            col_pos_emb[i, ::2] = np.sin(i/10000.0**(tmp[::2]/float(self.embed_dim)))
            col_pos_emb[i, 1::2] = np.cos(i/10000.0**(tmp[1::2]/float(self.embed_dim)))

        row_pos_emb = np.tile(np.expand_dims(row_pos_emb, 1), (1, self.max_cols, 1))
        col_pos_emb = np.tile(np.expand_dims(col_pos_emb, 0), (self.max_rows, 1, 1))
        pos_emb = np.concatenate([row_pos_emb, col_pos_emb], axis=-1)
        self.pos_emb = K.constant(np.reshape(pos_emb, (self.max_rows*self.max_cols, self.embed_dim*2)))

        super(TokenAndPositionEmbedding, self).build(input_shape)


# classification = "Negtivate"
# K.set_learning_phase(1)  # set learning phase
# path = "train_prediction_v2/" + classification
# 需根据自己情况修改1.训练好的模型路径和图像路径
# weight_file_dir = 'output1/head_train_multi_scale_transformer_4.h5'


model_path = "./train_prediction_v2/dataset1_train_multi_scale_transformer_4_0.h5"
model = keras.models.load_model(model_path, custom_objects={'stack': K.stack,
                                                          'AttentionLayer': AttentionLayer,
                                                          'TokenAndPositionEmbedding': TokenAndPositionEmbedding})

def check_file(filename):
    if os.path.exists(filename):
        shutil.rmtree(filename)

    path = filename

    if os.path.exists(path):
        print('exist')
    else:
        os.mkdir(path)


def cam(filename, filepath):
    # if os.path.exists("CAM_v2/"+ classification + "/" + filepath):
    #     return
    check_file(filepath)
    if os.path.exists(filepath)==False:
        os.mkdir(filepath)
    predict_csv=open(filename[0].split("_cell")[0]+".csv","w")
    predict_csv.write("file,pred\n")
    for j in filename:
        img_path = j
        x = Image.open(img_path)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x/255.0)
        predict_csv.write(img_path.split("/")[-1]+","+str(pred[0])+",\n")
        class_idx = np.argmax(pred[0])
        # print(model.summary())
        class_output = model.output[:, class_idx]
        # 需根据自己情况修改2. 把block5_conv3改成自己模型最后一层卷积层的名字
        last_conv_layer = model.get_layer("block2_conv2")

        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        ##需根据自己情况修改3. 512是我最后一层卷基层的通道数，根据自己情况修改
        for i in range(64):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
        img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img_to_array(image)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img0, 0.6, heatmap, 0.4, 0)
        #img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        print(filepath + "/" + j.split("\\")[-1])
        cv2.imwrite(filepath + "/" + j.split("\\")[-1], superimposed_img)
    # K.clear_session()

path=sys.argv[1]
gpu=sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
if "Positivate" in path:
    classification="Positivate"
elif "Negtivate" in path:
    classification="Negtivate"


filename = glob.glob(path+ "/*.jpg")
print(filename)
print("train_data_20220824/CAM/"+classification+"/"+path.split("/")[-2]+"/"+path.split("/")[-1]+"/")
print("\n\n\n\n\n\n\n\n\n\n")
cam(filename, "train_data_20220824/CAM/"+classification+"/"+path.split("/")[-2]+"/"+path.split("/")[-1]+"/")