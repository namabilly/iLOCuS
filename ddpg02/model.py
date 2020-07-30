import tensorflow as tf
from keras.models import Model
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Dropout, Conv2DTranspose,
                          Lambda, Concatenate, Reshape, LeakyReLU)
from keras.layers import merge, concatenate


def create_critic(look_back_steps, input_shape, num_actions, model_name='Critic'):
    # critic, value network --- same with q network
    '''

    :param look_back_steps:
    :param input_shape:
    :param num_actions:
    :param model_name:
    Input: state + action
    Output: Q(state, action)
    :return:
    '''
    with tf.name_scope(model_name):

        # input_img = Input(shape=(look_back_steps + 4,) + input_shape)  # state + action 4
        # State = Input(input_shape)
        input_img = Input(shape=(look_back_steps + 3,) + input_shape)  # state
        Action = Input(shape=[25])
        deconv1 = Conv2DTranspose(32, (5, 5), strides=(2, 2),
                                  input_shape=[look_back_steps + 4, input_shape[0], input_shape[1]],
                                  data_format='channels_first')(input_img)
        deconv1 = LeakyReLU(alpha=0.2)(deconv1)
        deconv2 = Conv2DTranspose(128, (5, 5), strides=(2, 2),
                                  input_shape=[look_back_steps + 4, input_shape[0], input_shape[1]],
                                  data_format='channels_first')(deconv1)
        deconv2 = LeakyReLU(alpha=0.2)(deconv2)
        conv1 = Convolution2D(64, (5, 5), data_format='channels_first', strides=(2, 2), padding='valid')(deconv2)
        conv1 = LeakyReLU(alpha=0.2)(conv1)

        flat = Flatten()(conv1)
        flat1 = Dense(25)(flat)

        #flat2 = Flatten()(Action)
        S_A = concatenate([flat1, Action], axis=-1)

        full = Dense(25)(S_A)

        out = LeakyReLU(alpha=0.2)(full)
        out = Dense(1)(S_A)
        # action = LeakyReLU(alpha=0.2)(full)
        model = Model(input=[input_img, Action], output=out)
    return model, input_img, Action


def create_actor(look_back_steps, input_shape, num_actions, model_name='Actor'):
    '''
    # input state， ouput action (action map)
    :param look_back_steps:
    :param input_shape:
    :param num_actions:
    :param model_name:
    :return:

    '''
    with tf.name_scope(model_name):
        input_img = Input(shape=(look_back_steps + 3,) + input_shape)
        deconv1 = Conv2DTranspose(32, (5, 5), strides=(2, 2),
                                  input_shape=[look_back_steps + 4, input_shape[0], input_shape[1]],
                                  data_format='channels_first')(input_img)
        deconv1 = LeakyReLU(alpha=0.2)(deconv1)
        deconv2 = Conv2DTranspose(128, (5, 5), strides=(2, 2),
                                  input_shape=[look_back_steps + 4, input_shape[0], input_shape[1]],
                                  data_format='channels_first')(deconv1)
        deconv2 = LeakyReLU(alpha=0.2)(deconv2)
        conv1 = Convolution2D(64, (5, 5), data_format='channels_first', strides=(2, 2), padding='valid')(deconv2)
        conv1 = LeakyReLU(alpha=0.2)(conv1)

        flat = Flatten()(conv1)
        full = Dense(25)(flat)
        # out = LeakyReLU(alpha=0.2)(full)
        out = Dense(25, activation="sigmoid")(full)
        out = Lambda(lambda i: i * num_actions)(out)
        model = Model(input=input_img, output=out)
    return model, input_img


def embedding(input_placeholder, input_shape, embedding_dim, layer_name):
    with tf.name_scope(layer_name):
        # input_placeholder shape: (batch, 1, 15, 15)
        reshaped = Reshape(target_shape=(1,) + input_shape)(input_placeholder)
        conv1 = Convolution2D(4, (3, 3), data_format='channels_first', strides=(1, 1), padding='valid')(reshaped)
        conv1 = LeakyReLU(alpha=0.1)(conv1)
        # conv1 shape: (batch, 4, 7, 7)

        return conv1
