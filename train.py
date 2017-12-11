import tensorflow as tf
import numpy as np
from model import model
from datalab import DataLabTrain, DataLabTest
import os


def train():
    Z, model_params = model()
    Y_hat = tf.sigmoid(Z)

    X = model_params['input_layer']
    Y_true_ = tf.placeholder(dtype=tf.float32, shape=None)
    Y = tf.expand_dims(Y_true_, 1)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z, labels=Y))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            train_gen = DataLabTrain('./datasets/train_set/').generator()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            ix = 0
            for X_true, Y_true in train_gen:
                ix += 1
                if ix % 10 == 0:
                    l, y = sess.run([loss, Y_hat], feed_dict={X:X_true, Y:Y_true})
                    acc = np.mean(y.astype('int32') == Y_true.astype('int32'))
                    print('epoch: ' + str(ix) + ' loss: ' + str(l) + ' accuracy: ' + str(acc))
                else:
                    sess.run([train_step], feed_dict={X: X_true, Y_true_: Y_true})

                if ix % 100 == 0:
                    path = './models/model' + (str(ix))
                    os.makedirs(path)
                    saver.save(sess, path + '/model.ckpt')

                if ix == 1000:
                    break
        finally:
            sess.close()


if __name__ == '__main__':
    train()