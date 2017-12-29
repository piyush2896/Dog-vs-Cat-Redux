import tensorflow as tf
import numpy as np
from datalab import DataLabTest
from model import model
import matplotlib.pyplot as plt
from make_file import make_sub


def predict(model_path, batch_size):
    Z, model_params = model()
    Y_hat = tf.sigmoid(Z)
    X = model_params['input_layer']
    saver = tf.train.Saver()
    test_gen = DataLabTest('./datasets/test_set/').generator()
    Y = []
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for i in range(12500//batch_size+1):
            y = sess.run(Y_hat, feed_dict={X: next(test_gen)})
            y = 1 - y
            print(y.shape, end='\t')
            Y.append(y)
            print(len(Y), end='\r')
    Y = np.concatenate(Y)
    print(Y.shape)
    return Y


if __name__ == '__main__':
    Y = predict('./models/dog-vs-cat.ckpt', 16)
    np.save('out.npy', Y)
    make_sub('sub_1.csv')