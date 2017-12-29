import numpy as np
import cv2
import os


def save_to_file(file, X, Y=None):
    X = np.asarray(X)
    if Y == None:
        np.save(file, X)
    else:
        np.savez(file, X, Y)


def make_batches_and_save_helper(imgs_list, path, out_path, batch_type, img_shape, batch_size):
    batch_no = 0
    new_batch = []
    if batch_type == 'train' or batch_type == 'dev':
        Y = []

    for ix, img_name in enumerate(imgs_list):
        print('adding ' + img_name + ' in batch: ' + batch_type + '_' + str(batch_no))
        new_batch.append(cv2.resize(cv2.imread(path+'/'+img_name), img_shape))

        if batch_type == 'train' or batch_type == 'dev':
            Y.append([0, 1] if img_name.split('.')[0] == 'dog' else [1, 0])

        if (ix + 1) % batch_size == 0:

            if batch_type == 'train' or batch_type == 'dev':
                save_to_file(out_path + str(batch_no) + '.npz', new_batch, Y)
                Y = []
            else:
                save_to_file(out_path + str(batch_no) + '.npy', new_batch)
            batch_no += 1
            new_batch = []

    if len(new_batch) != 0:
        if batch_type == 'train' or batch_type == 'dev':
            save_to_file(out_path + str(batch_no) + '.npz', new_batch, Y)
        else:
            save_to_file(out_path + str(batch_no) + '.npy', new_batch)

def make_batches_and_save(path, output_path=None, batch_type='train',
                          batch_size=32, img_shape=(64, 64), shuffle=True,
                          n_dev_set=None, dev_set_out=None):
    if output_path == None:
        output_path = path

    output_path = output_path + '/' + batch_type + '_'
    imgs_list = os.listdir(path)
    if batch_type != 'train':
        imgs_list = sorted([int(x.split('.')[0]) for x in imgs_list])
        imgs_list = [str(x) + '.jpg' for x in imgs_list]
        make_batches_and_save_helper(imgs_list, path, output_path, 'test', img_shape, batch_size)
    if shuffle:
        np.random.shuffle(imgs_list)
    if dev_set_out != None:
        dev_set_out = dev_set_out + '/dev_set_'
        val_list = imgs_list[len(imgs_list) - n_dev_set:]
        imgs_list = imgs_list[:len(imgs_list)-n_dev_set]
        make_batches_and_save_helper(imgs_list, path, output_path, 'train', img_shape, batch_size)
        make_batches_and_save_helper(val_list, path, dev_set_out, 'dev', img_shape, batch_size//2)


if __name__ == '__main__':
    make_batches_and_save('./datasets/train', './datasets/train_set',
                          batch_size=16, n_dev_set=2000, dev_set_out='./datasets/dev_set',
                          img_shape=(64, 64))
    make_batches_and_save('./datasets/test', './datasets/test_set',
                          batch_type='test', batch_size=16, shuffle=False,
                          img_shape=(64, 64))