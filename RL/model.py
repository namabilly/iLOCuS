import tensorflow as tf
from keras.models import Model
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Dropout,
                          Lambda, Concatenate, Reshape, LeakyReLU)


def create_model(look_back_steps, input_shape, num_actions, model_name='q_network'):
    with tf.name_scope(model_name):
        input_img = Input(shape = (look_back_steps + 4,) + input_shape) 
        # Input shape = (batch, look_back_steps + 5, 84, 84)
        
        embeddings = []
        for i in range(look_back_steps + 4):
            ch_i = Lambda(lambda x: x[:,i,:,:])(input_img)
            embeddings.append(embedding(ch_i, input_shape, 128, 'embed_'+str(i)))
        
        embed_feat = Concatenate(axis=1)(embeddings)
        
        conv1 = Convolution2D(64, (3,3), data_format='channels_first', strides=(1,1), padding='valid')(embed_feat)
        conv1 = LeakyReLU(alpha=0.1)(conv1)
        # (batch, 32, 5, 5)

        conv2 = Convolution2D(128, (1,1), data_format='channels_first', strides=(1,1), padding='valid')(conv1)
        conv2 = LeakyReLU(alpha=0.1)(conv2)
        # (batch, 128, 3, 3)

        flat = Flatten()(conv2)
        full = Dense(256)(flat)
        full = Activation('relu')(full)
        out = Dense(num_actions)(full) # output layer has node number = num_actions
        model = Model(input = input_img, output = out)
    return model

def embedding(input_placeholder, input_shape, embedding_dim, layer_name):
    with tf.name_scope(layer_name):
        # input_placeholder shape: (batch, 1, 15, 15)
        reshaped = Reshape(target_shape=(1,)+input_shape)(input_placeholder)
        conv1 = Convolution2D(4, (3,3), data_format='channels_first', strides=(1,1), padding='valid')(reshaped)
        conv1 = LeakyReLU(alpha=0.1)(conv1)
        # conv1 shape: (batch, 4, 7, 7)

        return conv1
