import tensorflow as tf
import numpy as np
from datalab import DataLabTest
from model import model
import matplotlib.pyplot as plt
from make_file import make_sub


def predict(model_path):
    Z, model_params = model()
    Y_hat = tf.sigmoid(Z)
    X = model_params['input_layer']
    saver = tf.train.Saver()
    test_gen = DataLabTest('./datasets/test_set/').generator()
    Y = []
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for i in range(782):
            y = sess.run(Y_hat, feed_dict={X: next(test_gen)})
            y[y < 0.5] = 1 - y[y < 0.5]
            Y.append(y)
    Y = np.concatenate(Y)
    print(Y.shape)
    return Y


if __name__ == '__main__':
    Y = predict('./models/model1000/model.ckpt')
    np.save('out.npy', Y)
    make_sub('sub_1.csv')