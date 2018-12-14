import os
import numpy as np


dirpath = os.path.dirname(__file__)
mnist_filepath = os.path.join(dirpath, 'mnist', 'mnist.npz')
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


