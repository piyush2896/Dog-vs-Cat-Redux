import tensorflow as tf
import numpy as np
import pprint


pp = pprint.PrettyPrinter(indent=4)


def conv2D(X, W, stage, block, 
           strides=[2, 2], padding='VALID'):
    name = 'conv_' + str(stage) + '_' + block
    strides = [1, strides[0], strides[1], 1]
    return tf.nn.conv2d(X, W, strides=strides, padding=padding, name=name)


def get_variable(shape, name):
    return tf.get_variable(name, shape=shape)


def flatten(tensor):
    return tf.contrib.layers.flatten(tensor)


def conv_layer(A_prev, n_C_prev, n_C, stage, block,
               padding='SAME', strides=(1, 1)):
    name = 'conv_' + str(stage) + '_' + block
    W = get_variable((3, 3, n_C_prev, n_C), name=name + '_W')
    b = get_variable((1, 1, 1, n_C), name=name + '_b')
    C = conv2D(A_prev, W, stage, block, padding=padding, strides=strides)
    Z = tf.add(C, b, name=name + '_Z')
    A = tf.nn.relu(Z, name='activation_' + str(stage) + block)
    conv_params = {'W': W, 'b': b, 'Z': Z}
    return A, conv_params


def dense_layer_wo_non_lin(A_prev, in_prev, out, stage, block):
    name = 'dense_' + str(stage) + '_' + block
    W = get_variable((in_prev, out), name=name + '_W')
    b = get_variable((1, out), name=name + '_b')
    Z = tf.add(tf.matmul(A_prev, W), b, name=name + '_Z')
    dense_params = {'W': W, 'b': b}
    return Z, dense_params


def dense_layer(A_prev, in_prev, out, stage, block):
    Z, dense_params = dense_layer_wo_non_lin(A_prev, in_prev,
                                             out, stage, block)
    A = tf.nn.relu(Z, name='activation_' + str(stage) + block)
    dense_params['Z'] = Z
    return A, dense_params


def model():
    X = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64], name='input_layer')
    X_ = tf.expand_dims(X, axis=3)
    params_dict = {}
    params_dict['input_layer'] = X
    params_dict['expand'] = X_

    # layer 1
    A1, params_dict['conv_1a'] = conv_layer(X_, 1, 32, 1, 'a', padding='SAME', strides=(1, 1))
    params_dict['conv_1a_out'] = A1

    A2 = tf.nn.max_pool(A1, ksize=(1, 5, 5, 1), strides=(1, 5, 5, 1),
                        padding='SAME', name='pool_1_a')
    params_dict['pool_1_a_out'] = A2

    # layer 2
    A3, params_dict['conv_1b'] = conv_layer(A2, 32, 64, 1, 'b')
    params_dict['conv_1b_out'] = A3

    A4 = tf.nn.max_pool(A3, ksize=(1, 5, 5, 1), strides=(1, 5, 5, 1),
                        padding='SAME', name='pool_1_b')
    params_dict['pool_1_b_out'] = A4

    # layer 3
    A5, params_dict['conv_1c'] = conv_layer(A4, 64, 128, 1, 'c')
    params_dict['conv_1c_out'] = A5
    
    A6 = tf.nn.max_pool(A5, ksize=(1, 5, 5, 1), strides=(1, 5, 5, 1),
                        padding='SAME', name='pool_1_c')
    params_dict['pool_1_c_out'] = A6

    # layer 4
    A7, params_dict['conv_1d'] = conv_layer(A6, 128, 64, 1, 'd')
    params_dict['conv_1d_out'] = A7

    # layer 3
    A8, params_dict['conv_1e'] = conv_layer(A7, 64, 32, 1, 'e')
    params_dict['conv_1e_out'] = A8

    A9 = tf.nn.max_pool(A8, ksize=(1, 5, 5, 1), strides=(1, 5, 5, 1),
                        padding='SAME', name='pool_1_e')
    params_dict['pool_1_e_out'] = A9

    # layer 5
    A10 = flatten(A9)
    params_dict['flatten_out'] = A10

    # layer 6
    A11, params_dict['dense_2a'] = dense_layer(A10, 1*1*32, 1024, 2, 'a')
    A12 = tf.nn.dropout(A11, keep_prob=0.8)
    params_dict['dense_2a_out'] = A11
    params_dict['dropout_2a'] = A12

    # layer 7
    A13, params_dict['dense_2b'] = dense_layer(A12, 1024, 1024, 2, 'b')
    params_dict['dense_2b_out'] = A13

    # layer 8
    Z, params_dict['dense_2c'] = dense_layer_wo_non_lin(A12, 1024, 1, 2, 'c')
    return Z, params_dict


if __name__ == '__main__':
    pp.pprint(model())