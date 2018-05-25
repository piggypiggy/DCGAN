import tensorflow as tf
import math
import numpy as np
import cv2 as cv

def deconv2d(input_, output_shape,
    filter_h=5, filter_w=5, strides_h=2, strides_w=2, stddev=0.02, padding='SAME', name=None):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [filter_h, filter_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.truncated_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, strides_h, strides_w, 1], padding=padding)

        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, b)
    return deconv

def conv2d(input_, output_channels,
    filter_h=5, filter_w=5, stride_h=2, stride_w=2, stddev=0.02, padding='SAME', name=None):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [filter_h, filter_w, input_.get_shape()[-1], output_channels],
                initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_, w, strides=[1, stride_h, stride_w, 1], padding=padding)

        b = tf.get_variable('b', output_channels, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
    return conv

def dense(input_, output_size, stddev=0.02, bias_init=0.0, name=None):
    input_size = input_.get_shape().as_list()[1]
    with tf.variable_scope(name):
        matrix = tf.get_variable("matrix", [input_size, output_size], tf.float32,
                    tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(input_, matrix) + bias

def batch_normalization(input_, name=None):
    return tf.layers.batch_normalization(input_, name=name)

def concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape() # [y_shape[0], 1, 1, y_shape[3]]
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def lrelu(input_, leak, name=None):
    return tf.maximum(input_, leak * input_)

class GAN():
    def __init__(self, height, width, channel):
        self.height = height
        self.width = width
        self.channel = channel # 图像通道数
        # self.batch_size = batch_size
        self.layers = 4
        self.num_filter = 32
        self.calc()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def calc_output_shape(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def calc(self):
        self.h = [0] * self.layers
        self.w = [0] * self.layers
        self.c = [0] * self.layers
        self.h[0], self.w[0], self.c[0] = self.height, self.width, int(0.5 * self.num_filter)
        for i in range(1, self.layers):
            self.h[i] = self.calc_output_shape(self.h[i-1], 2) 
            self.w[i] = self.calc_output_shape(self.w[i-1], 2)
            self.c[i] = int(self.c[i-1] * 2)
        self.c[0] = self.channel
        self.h.reverse() # h递增
        self.w.reverse()
        self.c.reverse() # c递减
        self.dense_size = self.h[1] * self.w[1] * self.c[1]

    def generator(self, z, label):
        with tf.variable_scope('generator'):
            L = tf.reshape(label, [self.batch_size, 1, 1, 10])
            net = tf.concat([z, label], 1)

            net = dense(z, self.dense_size, name='dense')
            net = batch_normalization(net, name='bn1')
            net = tf.nn.relu(net)
            net = tf.reshape(net, [self.batch_size, self.h[1], self.w[1], self.c[1]])
            net = concat(net, L)

            net = deconv2d(net, [self.batch_size, self.h[2], self.w[2], self.c[2]], name='deconv2')
            net = batch_normalization(net, name='bn2')
            net = tf.nn.relu(net)
            net = concat(net, L)

            net = deconv2d(net, [self.batch_size, self.h[3], self.w[3], self.c[3]], name='deconv3')
            net = tf.nn.tanh(net)
        return net

    def discriminator(self, img, label, reuse):
        with tf.variable_scope('discriminator') as scope: 
            if reuse:
                scope.reuse_variables()

            net = conv2d(img, self.c[2], name='conv1')
            net = batch_normalization(net, name='bn1')
            net = lrelu(net, leak=0.2)

            net = conv2d(img, self.c[1], name='conv2')
            net = batch_normalization(net, name='bn2')
            net = lrelu(net, leak=0.2)

            net = conv2d(net, self.c[0], name='conv3')
            net = batch_normalization(net, name='bn3')
            net = lrelu(net, leak=0.2)
            
            shape = net.get_shape()
            net = tf.reshape(net, [-1, shape[1]*shape[2]*shape[3]])
            net = dense(net, 10, name='dense')
            net = tf.nn.softmax(net)
            # net = tf.reduce_sum(label * net, 1)
        return net

if __name__ == '__main__':
    z = tf.constant(1.0, shape=[5, 100])
    p = tf.constant(1.0, shape=[5, 10])
    g = GAN(28, 28, 1, 5)
    r = g.generator(z,p)
    print(r.get_shape())
    

