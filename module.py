from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def feature_extraction_network(inputs, reuse=False):
    '''
    input : 64,64,1
    conv_1 : 64,64,64
    conv_2 : 64,64,64
    pool_1 : 32,32,64
    conv_3 : 32,32,128
    pool_2 : 16,16,128
    conv_4 : 16,16,256
    pool_3 : 8,8,256
    feature layer : 512
    '''
    with tf.variable_scope("feature_extraction_network"):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer=tf.truncated_normal_initializer(stddev=0.01)): 
            #inputs = slim.flatten(inputs)
            net = slim.conv2d(inputs=inputs, num_outputs=64, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv1') 
            net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv2') 
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
            net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv3') 
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
            net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], normalizer_fn=slim.batch_norm, scope='conv4') 
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool3')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 512, activation_fn=tf.nn.tanh)
            net = tf.expand_dims(net, axis=2)
        return net

def prediction_network(inputs, reuse=False):

    with tf.variable_scope("prediction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        inputs = slim.flatten(inputs)
        net_a = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)

        net_a = slim.fully_connected(net_a, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(net_b, 512, activation_fn=tf.nn.tanh)

        spectrum_a = slim.fully_connected(net_a, 101, activation_fn=tf.nn.tanh)
        spectrum_b = slim.fully_connected(net_b, 101, activation_fn=tf.nn.tanh)

        spectra = tf.concat([spectrum_a, spectrum_b], axis=1)
        spectra = tf.expand_dims(spectra, axis=2)
        return spectra


def recognition_network(feature, spectra, latent_dims, reuse=False):

    with tf.variable_scope("recognition_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        spectra = slim.flatten(spectra)
        feature = slim.flatten(feature)
        spectrum_a, spectrum_b = spectra[:,:101], spectra[:,101:]
        net = tf.concat([feature, spectrum_a, spectrum_b], axis=1)
        mean = slim.fully_connected(net, latent_dims, activation_fn=tf.nn.tanh)
        covariance = slim.fully_connected(net, latent_dims, activation_fn=tf.nn.tanh)

        return mean, covariance

def reconstruction_network(spectra, latent_variables, reuse=False):

    with tf.variable_scope("reconstruction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        spectra = slim.flatten(spectra)
        spectrum_a, spectrum_b = spectra[:,:101], spectra[:,101:]
        net = tf.concat([spectrum_a, spectrum_b, latent_variables], axis=1)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.tanh) # 512
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.tanh) # 512
        net = slim.fully_connected(net, 8*8*256, activation_fn=tf.nn.tanh) # 8*8*256
        net = tf.reshape(net, [-1,8,8,256])
        net = slim.conv2d_transpose(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.leaky_relu) # (16, 16, 128)
        net = slim.conv2d_transpose(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.leaky_relu) # (32, 32, 64)
        net = slim.conv2d_transpose(inputs=net, num_outputs=1, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.tanh)  # (64, 64, 1)

        return net

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
