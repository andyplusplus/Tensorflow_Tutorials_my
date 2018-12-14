# Load Data  In [3]:
from mnist import MNIST # package: python-mnist

def get_mnist(path="datasets/mnist/"):
    data = MNIST(path)

    print("Size of:")
    print("- Training-set:\t\t{}".format(data.num_train))
    print("- Validation-set:\t{}".format(data.num_val))
    print("- Test-set:\t\t{}".format(data.num_test))
    return data
