
import tensorflow as tf
from datetime import timedelta
import time
start_time = time.time()
is_plot = False

from common.plot_helper import plot_images

from common.mnist_common import get_mnist
data = get_mnist()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2]:
if is_plot: print("TensorFlow Version", tf.__version__)

