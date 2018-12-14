import os

from datasets import input_data

dirpath = os.path.dirname(__file__)
mnist_filepath = os.path.join(dirpath, 'mnist')

mnist_obj = None

def get_mnist_gzs():
    global mnist_obj
    global mnist_filepath
    if not mnist_obj:
        mnist_obj = input_data.read_data_sets(mnist_filepath)
    return mnist_obj