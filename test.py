import tensorflow as tf
import numpy as np
import cv2 as cv
import model
import os
import argparse

def load_model(sess, path):
    ckpt_path = os.path.join(path, 'checkpoint')
    if (os.path.exists(ckpt_path)):
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(path))
        print('model restored')
        return True
    else:
        print('model not exists')
        return False

def infer(args):
    which_num = args.num
    height, width, channel= 28, 28, 1
    z_size = args.nd
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        z = tf.placeholder(tf.float32, [None, z_size])
        label = tf.placeholder(tf.float32, [None, 10]) # 0~9
        gan = model.GAN(height, width, channel)
        # prob = gan.discriminator(img, label, reuse=False)
        if (which_num > -1): # 0~9中的某个数
            batch_size = 1
            gan.set_batch_size(batch_size)
            img = gan.generator(z, label)
            if (load_model(sess, args.model_dir)):
                batch_z = np.random.normal(0, 1, [batch_size, z_size]).astype(np.float_)
                label_data = np.zeros((batch_size, 10), dtype=np.float)
                label_data[0, which_num] = 1.0
                images = sess.run(img, feed_dict={z: batch_z, label: label_data}) # [1, height, width, channel] 
                image = images[0, :, :, :] * 255
                image = image.astype('uint8')
                cv.imwrite(str(which_num)+'.jpg', image)
                cv.imshow('infer', image)
                cv.waitKey(0)
        elif (which_num == -1): # 每个数字生成10张图片
            batch_size = 100 # 10*10
            gan.set_batch_size(batch_size)
            img = gan.generator(z, label)
            if (load_model(sess, args.model_dir)):
                batch_z = np.random.normal(0, 1, [batch_size, z_size]).astype(np.float_)
                label_data = np.zeros((batch_size, 10), dtype=np.float)
                for i in range(batch_size):
                    label_data[i, i//10] = 1.0
                images = sess.run(img, feed_dict={z: batch_z, label: label_data})
                image = np.zeros((28*10,28*10,1))
                for row in range(10):
                    for col in range(10):
                        image[28*row:28*(row+1), 28*col:28*(col+1)] = images[10*row+col]
                image = image * 255
                image = image.astype('uint8')
                cv.imwrite('all.jpg', image)
                cv.namedWindow('images', cv.WINDOW_NORMAL)
                cv.imshow('images', image)
                cv.waitKey()



                


  
