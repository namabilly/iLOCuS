import tensorflow as tf
from keras.models import Model
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Dropout,
                          Permute, Concatenate)

def create_model(look_back_steps, input_shape, num_actions, model_name='q_network'):
    with tf.name_scope(model_name):
        input_img = Input(shape = (look_back_steps + 5,) + input_shape) 
        # Input shape = (batch, look_back_steps + 5, 84, 84)

        embeddings = []
        for i in range(look_back_steps + 5):
            embeddings.append(embedding(input_img[:,i,:,:], 128, 'embed_'+str(i)))
        
        embed_feat = Concatenate(axis=1)(embeddings)
        full = Dense(256)(embed_feat)
        full = Activation('relu')(full)
        out = Dense(num_actions)(full) # output layer has node number = num_actions
        model = Model(input = input_img, output = out)
    return model

def embedding(input_placeholder, embedding_dim, layer_name):
    with tf.name_scope(layer_name):
        # input_placeholder shape: (batch, 1, 15, 15)

        conv1 = Convolution2D(16, (3,3), data_format='channels_first', stride=(2,2), padding='valid')(input_placeholder)
        conv1 = Activation('relu')(conv1)
        # conv1 shape: (batch, 1, 7, 7)

        conv2 = Convolution2D(32, (3,3), data_format='channels_first', stride=(2,2), padding='valid')(conv1)
        conv2 = Activation('relu')(conv2)
        # conv2 shape: (batch, 1, 3, 3)

        flat = Flatten()(conv2) # Flatten the convoluted hidden layers before full-connected layers
        full = Dense(embedding_dim)(flat) # embedding does not need activation
        # full shape: (batch, 128)
        return full