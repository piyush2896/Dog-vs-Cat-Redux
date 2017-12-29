import tensorflow as tf
import numpy as np
from model import model
from datalab import DataLabTrain, DataLabTest
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(n_iter):
    Z, model_params = model()
    Y_hat = tf.sigmoid(Z)

    X = model_params['input_layer']
    Y_true_ = tf.placeholder(dtype=tf.float32, shape=None)
    Y = tf.expand_dims(Y_true_, 1)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z, labels=Y))
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(n_iter):
                train_dl = DataLabTrain('./datasets/train_set/')
                train_gen = train_dl.generator()
                dev_dl = DataLabTrain('./datasets/dev_set/')
                dev_gen = dev_dl.generator()

                count = 0
                for X_true, Y_true in train_gen:
                    count += 1
                    sess.run(train_step, feed_dict={X:X_true, Y_true_: Y_true})
                    print('{}% completed'.format(round(count / train_dl.max_len * 100, 2)), end='\r')
                print()
                count = 0
                l = 0
                for X_true, Y_true in dev_gen:
                    count += 1
                    l += sess.run(loss, feed_dict={X:X_true, Y_true_: Y_true})
                print('Epoch: {}\tLoss: {}'.format(i, l / count), end='\r')
                print()
                saver.save(sess, './models/dog-vs-cat.ckpt')
                print("Model Saved")
        finally:
            sess.close()


if __name__ == '__main__':
    train(n_iter=100)