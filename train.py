import tensorflow as tf
import numpy as np
import cv2 as cv
import model
import os
import argparse
from data import DataProvider
from test import infer

def train(args):
    height, width, channel = 28, 28, 1
    batch_size = args.batch_size
    z_size = args.nd # 噪声维数
    real_img = tf.placeholder(tf.float32, [batch_size, height, width, channel], name='img') 
    z = tf.placeholder(tf.float32, [batch_size, z_size], name='z')
    label = tf.placeholder(tf.float32, [batch_size, 10], name='label') # 0~9

    gan = model.GAN(height, width, channel)
    gan.set_batch_size(batch_size)
    fake_img = gan.generator(z, label)
    real_result = gan.discriminator(real_img, label, reuse=False)
    fake_result = gan.discriminator(fake_img, label, reuse=True)
    real = tf.reduce_sum(label * real_result, 1)
    fake = tf.reduce_sum(label * fake_result, 1)
    d_loss = -tf.reduce_mean(tf.log(real) + tf.log(1. - fake))
    g_loss = -tf.reduce_mean(tf.log(fake))

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5) \
              .minimize(d_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5) \
              .minimize(g_loss, var_list=g_vars)

    data = DataProvider()
    train_num = data.get_train_num()
    batch_num = int(train_num / args.batch_size)

    saver = tf.train.Saver(max_to_keep=1)
    model_dir = args.model_dir
    if (not os.path.exists(model_dir)):
        os.mkdir(model_dir)

    accuracy_real = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, 1), tf.argmax(real_result, 1)), 'float'))
    accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, 1), tf.argmax(fake_result, 1)), 'float'))

    with tf.Session() as sess:
        counter = 0
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epoch):
            for batch in range(batch_num):
                counter += 1
                train_data, label_data = data.next_batch(batch_size)
                batch_z = np.random.normal(0, 1, [batch_size, z_size]).astype(np.float_)

                sess.run(d_optimizer, feed_dict={real_img: train_data, z: batch_z, label: label_data})
                sess.run(g_optimizer, feed_dict={z: batch_z, label: label_data})

                if (counter % 20 == 0):
                    dloss, gloss, ac_real, ac_fake = sess.run([d_loss, g_loss, accuracy_real, accuracy_fake], feed_dict={real_img: train_data, z: batch_z, label: label_data})
                    print('iter:', counter, 'd_loss:', dloss, 'g_loss:', gloss, 'ac_real:', ac_real, 'ac_fake:', ac_fake)
                if (counter % 200 == 0):
                    saver.save(sess, os.path.join(model_dir, 'model'))

