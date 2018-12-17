# # TensorFlow Tutorial #03-B
# # Layers API

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
import math

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

# ## Data Dimensions

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

# ### Helper-function for plotting images

from common.plot_helper import plot_images

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)

# ## TensorFlow Graph
# In[9]: # ## Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# In[13]: # ## PrettyTensor Implementation
if False:
    x_pretty = pt.wrap(x_image)
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.            conv2d(kernel=5, depth=16, name='layer_conv1').            max_pool(kernel=2, stride=2).            conv2d(kernel=5, depth=36, name='layer_conv2').            max_pool(kernel=2, stride=2).            flatten().            fully_connected(size=128, name='layer_fc1').            softmax_classifier(num_classes=num_classes, labels=y_true)











# In[14]: # ## Layers Implementation
net = x_image
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
layer_conv1 = net

net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
layer_conv2 = net
net = tf.layers.max_pooling2d(inputs=net,
                              pool_size=2, strides=2) # (?, 7, 7, 36)

net = tf.contrib.layers.flatten(net) # (?, 1764)
# net = tf.layers.flatten(net)
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)

# In[26]: # The output of the final fully-connected layer are sometimes called logits, so we have a convenience variable with that name.
logits = net
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# In[29]: # ### Loss-Function to be Optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# In[31]: # ### Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# In[34]: # ### Getting the Weights
for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(var)

# In[35]:
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable

# In[36]: # Using this helper-function we can retrieve the variables. These are TensorFlow objects. In order to get the contents of the variables, you must do something like: `contents = session.run(weights_conv1)` as demonstrated further below.
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')



# In[37]: # ## TensorFlow Run
session = tf.Session()
session.run(tf.global_variables_initializer())

# In[39]: # ### Helper-function to perform optimization iterations
train_batch_size = 64
total_iterations = 0
def optimize(num_iterations):
    global total_iterations
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
    total_iterations += num_iterations

def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls
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

# ### Helper-function for showing the performance
test_batch_size = 256
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,
                     y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    cls_true = data.test.cls
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

# In[45]: # ## Performance before any optimization
print_test_accuracy()

optimize(num_iterations=1)
print_test_accuracy()

optimize(num_iterations=99) # We already performed 1 iteration above.')
print_test_accuracy(show_example_errors=True)

optimize(num_iterations=900) # We performed 100 iterations above.')
print_test_accuracy(show_example_errors=True)

optimize(num_iterations=9000) # We performed 1000 iterations above.')
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ## Visualization of Weights and Layers

# ### Helper-function for plotting convolutional weights

# In[53]:
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

# ### Helper-function for plotting the output of a convolutional layer
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

# ### Input Images
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')
    if is_plot: plt.show()

# In[56]: # Plot an image from the test-set which will be used as an example below.
image1 = data.test.images[0]
plot_image(image1)

image2 = data.test.images[13]
plot_image(image2)

# ### Convolution Layer 1
plot_conv_weights(weights=weights_conv1)
plot_conv_layer(layer=layer_conv1, image=image1)
plot_conv_layer(layer=layer_conv1, image=image2)
plot_conv_weights(weights=weights_conv2, input_channel=0)
plot_conv_weights(weights=weights_conv2, input_channel=1)
plot_conv_layer(layer=layer_conv2, image=image1)
plot_conv_layer(layer=layer_conv2, image=image2)

# ### Close TensorFlow Session
session.close()
print_time_usage(start_time_global)


# ## Exercises
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# You may want to backup this Notebook before making any changes.
# * Change the activation function to sigmoid for some of the layers.
# * Can you find a simple way of changing the activation function for all the layers?
# * Add a dropout-layer after the fully-connected layer. If you want a different probability during training and testing then you will need a placeholder variable and set it in the feed-dict.
# * Plot the output of the max-pooling layers instead of the conv-layers.
# * Replace the 2x2 max-pooling layers with stride=2 in the convolutional layers. Is there a difference in classification accuracy? What if you optimize it again and again? The difference is random, so how would you measure if there really is a difference? What are the pros and cons of using max-pooling vs. stride in the conv-layer?
# * Change the parameters for the layers, e.g. the kernel, depth, size, etc. What is the difference in time usage and classification accuracy?
# * Add and remove some convolutional and fully-connected layers.
# * What is the simplest network you can design that still performs well?
# * Retrieve the bias-values for the convolutional layers and print them. See `get_weights_variable()` for inspiration.
# * Remake the program yourself without looking too much at this source-code.
# * Explain to a friend how the program works.

