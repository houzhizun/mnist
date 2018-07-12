import numpy as np
import tensorflow as tf
import input_data
from logisticRegression import LogisticRegression
from mlp import HiddenLayer
class ConvLayer(object):
    """
    A convolution layer
    """
    def __init__(self,inpt,filter_shape,strides=(1,1,1,1),
                 padding="SAME",activation=tf.nn.relu,bias_setting=True):
        """
        inpt: tf.Tensor, shape [n_examples, witdth, height, channels]
        filter_shape: list or tuple, [witdth, height. channels, filter_nums]
        strides: list or tuple, the step of filter
        padding:
        activation:
        bias_setting:
        """
        self.input = inpt
        self.W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),dtype=tf.float32)
        if bias_setting:
            self.b = tf.Variable(tf.truncated_normal(filter_shape[-1:],stddev=0.1),dtype=tf.float32)
        else:
            self.b = None
        conv_output = tf.nn.conv2d(self.input,filter=self.W,strides=strides,
                                   padding=padding)
        conv_output = conv_output + self.b if self.b is not None else conv_output
        # the output
        self.output = conv_output if activation is None else activation(conv_output)
        # the params
        self.params = [self.W,self.b] if self.b is not None else [self.W,]

class MaxPoolLayer(object):
    """pool layer"""
    def __init__(self,inpt,ksize=(1,2,2,1),strides=(1,2,2,1),padding="SAME"):
        self.input = inpt
        # the output
        self.output = tf.nn.max_pool(self.input,ksize=ksize,strides=strides,padding=padding)
        self.params = []
class FlattenLayer(object):
    def __init__(self,inpt,shape):
        self.input = inpt
        self.output = tf.reshape(self.input,shape=shape)
        self.params = []
class DropoutLayer(object):
    def __init__(self,inpt,keep_prob):
        self.keep_prob = tf.placeholder(tf.float32)
        self.input = inpt
        self.output = tf.nn.dropout(self.input,keep_prob=self.keep_prob)
        self.train_dicts = {self.keep_prob: keep_prob}
        self.pred_dicts = {self.keep_prob: 1.0}


































