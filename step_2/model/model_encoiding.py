#!usr/bin/env python
#coding:utf-8
"""
@author: Haidong Zhang
@contact: haidong_zhang14@yahoo.com
@time: 2021/1/9 15:28
@project: MalariaDetection
@description:
"""
from keras.layers import *      #For adding convolutional layer
from keras.layers import Dense, ZeroPadding2D, BatchNormalization,  Activation        #For adding layers to NN

from keras.models import *                  #for loading the model
from keras.backend import stack, squeeze, expand_dims
from utils import module
from keras import backend as K
from utils.attention import AttentionLayer
from model.transformer4 import LayerNormalization, TokenAndPositionEmbedding0
from keras_transformer.attention import MultiHeadSelfAttention, MultiHeadAttention

def multi_scale_transformer():
    size = 64
    rate = 0.1
    down_sampling_4 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down4'),
                                  BatchNormalization(), PReLU()])
    down_sampling_3 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down3'),
                                  BatchNormalization(), PReLU()])
    down_sampling_2 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down2'),
                                  BatchNormalization(), PReLU()])
    down_sampling_1 = Sequential([Conv2D(size, kernel_size=(1, 1), activation=None, padding='same', name='down1'),
                                  BatchNormalization(), PReLU()])
    # img_input, scale_outputs = module.base_encode()
    img_input, scale_outputs = module.vgg_encode()

    down4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(down_sampling_4(scale_outputs[4]))
    down3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(down_sampling_3(scale_outputs[3]))
    down2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(down_sampling_2(scale_outputs[2]))
    down1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(down_sampling_1(scale_outputs[1]))
    down0 = scale_outputs[0]

    # Transformer
    embed_dim = 64  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer
    print(f"embed_dim: {embed_dim}, num_heads: {num_heads}, ff_dim: {ff_dim}")

    rs_down0 = Reshape([32*32, embed_dim])(down0)
    rs_down4 = Reshape([32*32, embed_dim])(down4)
    rs_down3 = Reshape([32*32, embed_dim])(down3)
    rs_down2 = Reshape([32*32, embed_dim])(down2)
    rs_down1 = Reshape([32*32, embed_dim])(down1)


    if 0:
        print(f"+++++++++++++++++++ Position ++++++++++++++++++++++++++++++++")
        half_embed_dim = int(embed_dim/2)
        rs_down0 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down0)
        rs_down4 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down4)
        rs_down3 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down3)
        rs_down2 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down2)
        rs_down1 = TokenAndPositionEmbedding(32, 32, half_embed_dim)(rs_down1)
    # transformer encoding
    if 2:
        print(f"+++++++++++++++++++ Position ++++++++++++++++++++++++++++++++")
        half_embed_dim = int(embed_dim/2)
        rs_down0 = TokenAndPositionEmbedding0(32, 32, half_embed_dim)(rs_down0)
        rs_down4 = TokenAndPositionEmbedding0(32, 32, half_embed_dim)(rs_down4)
        rs_down3 = TokenAndPositionEmbedding0(32, 32, half_embed_dim)(rs_down3)
        rs_down2 = TokenAndPositionEmbedding0(32, 32, half_embed_dim)(rs_down2)
        rs_down1 = TokenAndPositionEmbedding0(32, 32, half_embed_dim)(rs_down1)


    ts_att_down3 = MultiHeadAttention(num_heads, use_masking=False)([rs_down3, rs_down4])
    ts_down3 = LayerNormalization()(Add()([rs_down3, Dropout(rate)(ts_att_down3)]))
    ts_down3 = LayerNormalization()(Add()([ts_down3, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down3))]))

    ts_att_down2 = MultiHeadAttention(num_heads, use_masking=False)([rs_down2, rs_down3])
    ts_down2 = LayerNormalization()(Add()([rs_down2, Dropout(rate)(ts_att_down2)]))
    ts_down2 = LayerNormalization()(Add()([ts_down2, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down2))]))

    ts_att_down1 = MultiHeadAttention(num_heads, use_masking=False)([rs_down1, rs_down2])
    ts_down1 = LayerNormalization()(Add()([rs_down1, Dropout(rate)(ts_att_down1)]))
    ts_down1 = LayerNormalization()(Add()([ts_down1, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down1))]))

    ts_att_down0 = MultiHeadAttention(num_heads, use_masking=False)([rs_down0, rs_down1])
    ts_down0 = LayerNormalization()(Add()([rs_down0, Dropout(rate)(ts_att_down0)]))
    ts_down0 = LayerNormalization()(Add()([ts_down0, Dropout(rate)(Dense(embed_dim, activation="relu")(ts_down0))]))

    hidden_states = [Flatten()(ts_down0), Flatten()(ts_down1), Flatten()(ts_down2),
                     Flatten()(ts_down3), Flatten()(rs_down4)]  #
    hidden_states = Lambda(lambda x: K.stack(x, axis=1))(hidden_states)
    hidden_size = 32 * 32 * size
    hidden_states = Reshape([5, hidden_size])(hidden_states)
    output = AttentionLayer(512, name='attention_2')(hidden_states)           # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    # output = concatenate([Flatten()(ts_down0), Flatten()(ts_down1), Flatten()(ts_down2),
    #                  Flatten()(ts_down3), Flatten()(rs_down4)])  #

    reps = Dense(256, activation="relu")(output)                            # 可以调整512为[64, 128, 256, 512, 1024, 2048]
    output = Dense(1, activation="sigmoid")(reps)
    model = module.model_compile(img_input, output)
    print(model.summary())

    return model


if __name__ == "__main__":
    multi_scale_transformer()
