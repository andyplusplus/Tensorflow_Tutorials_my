"""

from common.mnist_common import get_mnist
data = get_mnist()

"""
# Load Data  In [3]:
from common.file_operator import get_data_directory_mnist
from mnist import MNIST # package: python-mnist

def get_mnist(path=""):
    if(path == ""):
        path = get_data_directory_mnist()
    data = MNIST(path)

    print("Size of:")
    print("- Training-set:\t\t{}".format(data.num_train))
    print("- Validation-set:\t{}".format(data.num_val))
    print("- Test-set:\t\t{}".format(data.num_test))
    return data


from tensorflow.examples.tutorials.mnist import input_data
def get_mnist_4_prettyTensor(path=""):
    if(path == ""):
        path = get_data_directory_mnist()
    data = input_data.read_data_sets(path, one_hot=True)

    print("Size of:")
    print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Test-set:\t\t{}".format(len(data.test.labels)))
    print("- Validation-set:\t{}".format(len(data.validation.labels)))
    return data

