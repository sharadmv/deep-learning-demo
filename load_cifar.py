import numpy as np
import os
import cPickle as pickle
import glob

data_dir = "data"
data_dir_cifar10 = os.path.join(data_dir, "cifar-10-batches-py")
# data_dir_cifar100 = os.path.join(data_dir, "cifar-100-python")

class_names_cifar10 = np.load(os.path.join(data_dir_cifar10, "batches.meta"))
# class_names_cifar100 = np.load(os.path.join(data_dir_cifar100, "meta"))


def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _load_batch_cifar10(filename, dtype='float64'):
    """
    load a batch in the CIFAR-10 format
    """
    path = os.path.join(data_dir_cifar10, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0 # scale between [0, 1]
    labels = one_hot(batch['labels'], n=10) # convert labels to one-hot representation
    return data.astype(dtype), labels.astype(dtype)


def _grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)


def cifar10(dtype='float64', grayscale=True):
    # train
    x_train = []
    t_train = []
    for k in xrange(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    # test
    x_test, t_test = _load_batch_cifar10("test_batch", dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)

    x_train = x_train.reshape((x_train.shape[0], 32, 32))
    x_test = x_test.reshape((x_test.shape[0], 32, 32))
    return x_train[:, np.newaxis], t_train, x_test[:, np.newaxis], t_test
