import tensorflow as tf
from keras.models import Model
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Dropout, Conv2DTranspose,
                          Lambda, Concatenate, Reshape, LeakyReLU, ReLU)

# def create_model(look_back_steps, input_shape, num_actions, model_name='q_network'):
#     with tf.name_scope(model_name):
#         input_img = Input(shape = (look_back_steps + 3,) + input_shape)
#         # input_loc = Input(shape = [1] )
#         # input_loc = Lambda(lambda x: expand_dims(x, axis=1))(input_loc)
#         # print(input_loc.shape)
#         # Input shape = (batch, look_back_steps + 5, 84, 84)
#         # input_loc = input_img[:,-1,0,0]
#
#         # embeddings = []
#         # for i in range(look_back_steps + 4):
#         #     ch_i = Lambda(lambda x: x[:,i,:,:])(input_img)
#         #     embeddings.append(embedding(ch_i, input_shape, 128, 'embed_'+str(i)))
#
#         # embed_feat = Concatenate(axis=1)(embeddings)
#         deconv1 = Conv2DTranspose(32, (5, 5), strides=(2, 2),
#                                   input_shape=[look_back_steps + 4,input_shape[0],input_shape[1]],
#                                   data_format='channels_first')(input_img)
#         deconv1 = LeakyReLU(alpha=0.2)(deconv1)
#         deconv2 = Conv2DTranspose(128, (5, 5), strides=(2, 2),
#                                   input_shape=[look_back_steps + 4, input_shape[0], input_shape[1]],
#                                   data_format='channels_first')(deconv1)
#         deconv2 = LeakyReLU(alpha=0.2)(deconv2)
#         conv1 = Convolution2D(64, (5,5), data_format='channels_first', strides=(2,2), padding='valid')(deconv2)
#         conv1 = LeakyReLU(alpha=0.2)(conv1)
#         # (batch, 32, 5, 5)
#
#         conv2 = Convolution2D(128, (3,3), data_format='channels_first', strides=(1,1), padding='valid')(conv1)
#         conv2 = LeakyReLU(alpha=0.2)(conv2)
#         # (batch, 128, 3, 3)
#
#         flat = Flatten()(conv2)
#         full = Dense(250)(flat)
#         # full = LeakyReLU(alpha=0.2)(full)
#         #
#         # embed_feat = Concatenate(axis=1)([full, input_loc])
#         # print(embed_feat.shape)
#         # full = Dense(num_actions)(embed_feat) # output layer has node number = num_actions
#         out = LeakyReLU(alpha=0.2)(full)
#         model = Model(input = input_img, output = out)
#     return model


def create_model(look_back_steps, input_shape, num_actions, model_name='q_network'):
    with tf.name_scope(model_name):
        input_img = Input(shape=(look_back_steps + 3,) + input_shape)
        # Input shape = (batch, look_back_steps + 5, 84, 84)
        # embeddings = []
        # for i in range(look_back_steps + 4):
        #     ch_i = Lambda(lambda x: x[:,i,:,:])(input_img)
        #     embeddings.append(embedding(ch_i, input_shape, 128, 'embed_'+str(i)))

        # embed_feat = Concatenate(axis=1)(embeddings)
        # deconv1 = Conv2DTranspose(32, (5, 5), strides=(2, 2),
        #                           input_shape=[look_back_steps + 4, input_shape[0], input_shape[1]],
        #                           data_format='channels_first')(input_img)
        # deconv1 = LeakyReLU(alpha=0.2)(deconv1)
        # deconv2 = Conv2DTranspose(128, (5, 5), strides=(2, 2),
        #                           input_shape=[look_back_steps + 4, input_shape[0], input_shape[1]],
        #                           data_format='channels_first')(deconv1)
        # deconv2 = LeakyReLU(alpha=0.2)(deconv2)
        # conv1 = Convolution2D(64, (5, 5), data_format='channels_first', strides=(2, 2), padding='valid')(deconv2)
        # conv1 = LeakyReLU(alpha=0.2)(conv1)
        # # (batch, 32, 5, 5)
        #
        # conv2 = Convolution2D(128, (3, 3), data_format='channels_first', strides=(1, 1), padding='valid')(conv1)
        # conv2 = LeakyReLU(alpha=0.2)(conv2)
        # (batch, 128, 3, 3)

        flat = Flatten()(input_img)
        full = Dense(1280)(flat)
        full = LeakyReLU(alpha=0.2)(full)
        full = Dense(2560)(full)
        full = LeakyReLU(alpha=0.2)(full)
        out = Dense(25*num_actions)(full)  # output layer has node number = num_actions
        # out = LeakyReLU(alpha=0.2)(full)
        model = Model(input=input_img, output=out)
    return model

def embedding(input_placeholder, input_shape, embedding_dim, layer_name):
    with tf.name_scope(layer_name):
        # input_placeholder shape: (batch, 1, 15, 15)
        reshaped = Reshape(target_shape=(1,)+input_shape)(input_placeholder)
        conv1 = Convolution2D(4, (3,3), data_format='channels_first', strides=(1,1), padding='valid')(reshaped)
        conv1 = LeakyReLU(alpha=0.1)(conv1)
        # conv1 shape: (batch, 4, 7, 7)

        return conv1
