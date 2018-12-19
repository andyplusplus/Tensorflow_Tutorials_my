from common.data_reader import input_data
from common.file_operator import get_data_directory_mnist

mnist_directory = get_data_directory_mnist()
mnist_obj = None

def get_mnist_gzs():
    global mnist_obj
    global mnist_directory
    if not mnist_obj:
        mnist_obj = input_data.read_data_sets(mnist_directory)
    return mnist_obj