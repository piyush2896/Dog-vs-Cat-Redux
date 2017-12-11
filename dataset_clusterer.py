import numpy as np
import cv2
import os


def save_to_file(file, X, Y=None):
    X = np.asarray(X)
    if Y == None:
        Y = np.asarray([Y])
        np.save(file, X)
    else:
        np.savez(file, X, Y)


def make_batches_and_save(path, output_path=None, batch_type='train',
                          batch_size=16, img_shape=(64, 64), shuffle=True):
    if output_path == None:
        output_path = path

    output_path = output_path + '/' + batch_type + '_'
    imgs_list = os.listdir(path)
    np.random.shuffle(imgs_list)

    batch_no = 0
    new_batch = []
    if batch_type == 'train':
        Y = []

    for ix, img_name in enumerate(imgs_list):
        print('adding ' + img_name + 'in batch: ' + batch_type + '_' + str(batch_no))
        new_batch.append(cv2.resize(cv2.imread(path+'/'+img_name, cv2.IMREAD_GRAYSCALE), img_shape))

        if batch_type == 'train':
            Y.append(0 if img_name.split('.')[0] == 'dog' else 1)

        if (ix + 1) % 16 == 0:

            if batch_type == 'train':
                save_to_file(output_path + str(batch_no) + '.npz', new_batch, Y)
                Y = []
            else:
                save_to_file(output_path + str(batch_no) + '.npy', new_batch)
            batch_no += 1
            new_batch = []

    if batch_type == 'train':
        save_to_file(output_path + str(batch_no) + '.npz', new_batch, Y)
    else:
        save_to_file(output_path + str(batch_no) + '.npy', new_batch)


if __name__ == '__main__':
    make_batches_and_save('./datasets/train', './datasets/train_set')
    make_batches_and_save('./datasets/test', './datasets/test_set',
                          batch_type='test', batch_size=32, shuffle=False)