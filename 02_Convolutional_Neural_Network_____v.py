# # TensorFlow Tutorial #02
# # Convolutional Neural Network

# ## Introduction
# ## Flowchart
# ## Convolutional Layer

# In[1]: # ## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# In[3]: # ## Configuration of Neural Network
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# In[4]: # ## Load Data # In[5]: # The MNIST data-set has now been loaded and consists of 70.000 images and class-numbers for the images. The data-set is split into 3 mutually exclusive sub-sets. We will only use the training and test-sets in this tutorial.
from common.mnist_common import get_mnist
data = get_mnist()

img_size = data.img_size # The number of pixels in each dimension of an image.
img_size_flat = data.img_size_flat # The images are stored in one-dimensional arrays of this length.
img_shape = data.img_shape # Tuple with height and width of images used to reshape arrays.
num_classes = data.num_classes # Number of classes, one class for each of 10 digits.
num_channels = data.num_channels # Number of colour channels for the images: 1 channel for gray-scale.

# In[7]: # ### Helper-function for plotting images # Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
# In[8]: # ### Plot a few images to see if data is correct
from common.plot_helper import plot_images
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
plot_images(images=images, cls_true=cls_true)


# ## TensorFlow Graph # In[9]: # ### Helper-functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[11]: # ### Helper-function for creating a new Convolutional Layer
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)
    return layer, weights


# In[12]: # ### Helper-function for flattening a layer
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


# In[13]: # ### Helper-function for creating a new Fully-Connected Layer
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer





# In[14]: # ### Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)




# In[18]: # ### Convolutional Layer 1, (?, 14, 14, 16)
layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,# 1
                   filter_size=filter_size1,       # 5
                   num_filters=num_filters1,       #16
                   use_pooling=True)

# In[20]: # ### Convolutional Layer 2, (?, 7, 7, 36)
layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,  #16
                   filter_size=filter_size2,         # 5
                   num_filters=num_filters2,         #36
                   use_pooling=True)

# In[22]: # ### Flatten Layer    (?, 7, 7, 36),      1764 = 3 * 3 * 36
layer_flat, num_features = flatten_layer(layer_conv2)

# In[25]: # ### Fully-Connected Layer 1
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,  # 1764
                         num_outputs=fc_size,      #  128
                         use_relu=True)

# In[27]: # ### Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,       # 128
                         num_outputs=num_classes,  #  10
                         use_relu=False)

# In[31]: # ### Cost-function to be optimized # In[33]: # ### Optimization Method
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# ### Predicted Class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# In[34]: # ### Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ------------------------------------------------------------------------------------


# In[36]: # ## TensorFlow Run # In[37]: # ### Initialize variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# In[38]: # ### Helper-function to perform optimization iterations
train_batch_size = 64
total_iterations = 0
def optimize(num_iterations):
    global total_iterations
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# In[40]: # ### Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = data.x_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.y_test_cls[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# In[41]: # ### Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    cls_true = data.y_test_cls
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if is_plot: plt.show()


# In[42]: # ### Helper-function for showing the performance
test_batch_size = 256
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    num_test = data.num_test
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.x_test[i:j, :]
        labels = data.y_test[i:j, :]
        feed_dict = {x: images, y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    cls_true = data.y_test_cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)





# In[43]:Performance before any optimization
print_test_accuracy()

# In[44]:Performance after 1 optimization iteration
optimize(num_iterations=1)
print_test_accuracy()

# In[46]: # ## Performance after 100 optimization iterations
optimize(num_iterations=99)
print_test_accuracy(show_example_errors=True)

# In[48]: # ## Performance after 1000 optimization iterations
optimize(num_iterations=900)
print_test_accuracy(show_example_errors=True)

# In[50]: # ## Performance after 10,000 optimization iterations
optimize(num_iterations=9000)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)





# In[52]: # ## Visualization of Weights and Layers

# ### Helper-function for plotting convolutional weights

def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)

    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    if is_plot: plt.show()


# In[53]: # ### Helper-function for plotting the output of a convolutional layer
def plot_conv_layer(layer, image):
    feed_dict = {x: [image]}
    values = session.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    
    if is_plot: plt.show()


# In[54]: # ### Input Images
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')
    if is_plot: plt.show()




image1 = data.x_test[0]
plot_image(image1)

image2 = data.x_test[13]
plot_image(image2)

# In[57]: # ### Convolution Layer 1
plot_conv_weights(weights=weights_conv1)
plot_conv_layer(layer=layer_conv1, image=image1)
plot_conv_layer(layer=layer_conv1, image=image2)

# In[60]: # ### Convolution Layer 2
plot_conv_weights(weights=weights_conv2, input_channel=0)
plot_conv_weights(weights=weights_conv2, input_channel=1)
plot_conv_layer(layer=layer_conv2, image=image1)
plot_conv_layer(layer=layer_conv2, image=image2)

# ### Close TensorFlow Session
session.close()
print_time_usage(start_time_global)




# ## Conclusion
# We have seen that a Convolutional Neural Network works much better at recognizing hand-written digits than the simple linear model in Tutorial #01. The Convolutional Network gets a classification accuracy of about 99%, or even more if you make some adjustments, compared to only 91% for the simple linear model.
# However, the Convolutional Network is also much more complicated to implement, and it is not obvious from looking at the filter-weights why it works and why it sometimes fails.
# So we would like an easier way to program Convolutional Neural Networks and we would also like a better way of visualizing their inner workings.

# ## Exercises
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# You may want to backup this Notebook before making any changes.
# * Do you get the exact same results if you run the Notebook multiple times without changing any parameters? What are the sources of randomness?
# * Run another 10,000 optimization iterations. Are the results better?
# * Change the learning-rate for the optimizer.
# * Change the configuration of the layers, such as the number of convolutional filters, the size of those filters, the number of neurons in the fully-connected layer, etc.
# * Add a so-called drop-out layer after the fully-connected layer. Note that the drop-out probability should be zero when calculating the classification accuracy, so you will need a placeholder variable for this probability.
# * Change the order of ReLU and max-pooling in the convolutional layer. Does it calculate the same thing? What is the fastest way of computing it? How many calculations are saved? Does it also work for Sigmoid-functions and average-pooling?
# * Add one or more convolutional and fully-connected layers. Does it help performance?
# * What is the smallest possible configuration that still gives good results?
# * Try using ReLU in the last fully-connected layer. Does the performance change? Why?
# * Try not using pooling in the convolutional layers. Does it change the classification accuracy and training time?
# * Try using a 2x2 stride in the convolution instead of max-pooling? What is the difference?
# * Remake the program yourself without looking too much at this source-code.
# * Explain to a friend how the program works.

