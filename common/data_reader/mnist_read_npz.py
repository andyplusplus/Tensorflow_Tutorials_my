import os
import numpy as np

from common.file_operator import get_data_directory_mnist

dirpath = get_data_directory_mnist()
mnist_filepath = os.path.join(dirpath, 'mnist.npz')
x_train, y_train = None, None
x_test, y_test = None, None

def load_mnist_npz():
    global mnist_filepath
    global x_train, y_train, x_test, y_test
    if not x_train:
        f = np.load(mnist_filepath)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
    return (x_train, y_train), (x_test, y_test)


