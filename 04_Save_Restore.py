# # TensorFlow Tutorial #04
# # Save & Restore
# does not work because prettyTensor

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time=get_start_time()
is_plot = False
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

# PrettyTensor version:

# In[4]:
pt.__version__

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# ## Data Dimensions

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

# ### Helper-function for plotting images

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    if is_plot: plt.show()

# ### Plot a few images to see if data is correct
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)

# ## TensorFlow Graph
# ### Placeholder variables
# In[11]:
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# ### Neural Network
# In[15]:
x_pretty = pt.wrap(x_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# ### Getting the Weights
# In[17]:
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')
    return variable

# Using this helper-function we can retrieve the variables. These are TensorFlow objects. In order to get the contents of the variables, you must do something like: `contents = session.run(weights_conv1)` as demonstrated further below.
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# ### Optimization Method
# In[19]:
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# ### Performance Measures
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))






# ### Saver
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')

# ## TensorFlow Run
# In[27]:
session = tf.Session()
def init_variables():
    session.run(tf.global_variables_initializer())
init_variables()

# ### Helper-function to perform optimization iterations
train_batch_size = 64
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 1000
total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement
    start_time = time.time()

    for i in range(num_iterations):
        total_iterations += 1
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            acc_validation, _ = validation_accuracy()
            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                last_improvement = total_iterations
                saver.save(sess=session, save_path=save_path)
                improved_str = '*'
            else:
                improved_str = ''
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")
            break

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# ### Helper-function to plot example errors
# In[33]:
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# ### Helper-function to plot confusion matrix
# In[34]:
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

# ### Helper-functions for calculating classifications
batch_size = 256

def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    correct = (cls_true == cls_pred)
    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = data.test.images,
                       labels = data.test.labels,
                       cls_true = data.test.cls)

def predict_cls_validation():
    return predict_cls(images = data.validation.images,
                       labels = data.validation.labels,
                       cls_true = data.validation.cls)

# ### Helper-functions for the classification accuracy
# In[38]:
def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

# Calculate the classification accuracy on the validation-set.

# In[39]:
def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()
    
    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)

# ### Helper-function for showing the performance

# Function for printing the classification accuracy on the test-set.
# It takes a while to compute the classification for all the images in the test-set, that's why the results are re-used by calling the above functions directly from this function, so the classifications don't have to be recalculated by each function.

# In[40]:
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = cls_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# ### Helper-function for plotting convolutional weights

# In[41]:
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    if is_plot: plt.show()

# ## Performance before any optimization
# The accuracy on the test-set is very low because the model variables have only been initialized and not optimized at all, so it just classifies the images randomly.

# In[42]:
print_test_accuracy()

# The convolutional weights are random, but it can be difficult to see any difference from the optimized weights that are shown below. The mean and standard deviation is shown so we can see whether there is a difference.

# In[43]:
plot_conv_weights(weights=weights_conv1)

# ## Perform 10,000 optimization iterations
# We now perform 10,000 optimization iterations and abort the optimization if no improvement is found on the validation-set in 1000 iterations.
# An asterisk * is shown if the classification accuracy on the validation-set is an improvement.

# In[44]:
optimize(num_iterations=10000)

# In[45]:
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# The convolutional weights have now been optimized. Compare these to the random weights shown above. They appear to be almost identical. In fact, I first thought there was a bug in the program because the weights look identical before and after optimization.
# But try and save the images and compare them side-by-side (you can just right-click the image to save it). You will notice very small differences before and after optimization.
# The mean and standard deviation has also changed slightly, so the optimized weights must be different.

# In[46]:
plot_conv_weights(weights=weights_conv1)

# ## Initialize Variables Again
# Re-initialize all the variables of the neural network with random values.

# In[47]:
init_variables()

# This means the neural network classifies the images completely randomly again, so the classification accuracy is very poor because it is like random guesses.

# In[48]:
print_test_accuracy()

# The convolutional weights should now be different from the weights shown above.

# In[49]:
plot_conv_weights(weights=weights_conv1)

# ## Restore Best Variables
# Re-load all the variables that were saved to file during optimization.

# In[50]:
saver.restore(sess=session, save_path=save_path)

# The classification accuracy is high again when using the variables that were previously saved.
# Note that the classification accuracy may be slightly higher or lower than that reported above, because the variables in the file were chosen to maximize the classification accuracy on the validation-set, but the optimization actually continued for another 1000 iterations after saving those variables, so we are reporting the results for two slightly different sets of variables. Sometimes this leads to slightly better or worse performance on the test-set.

# In[51]:
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# The convolutional weights should be nearly identical to those shown above, although not completely identical because the weights shown above had 1000 optimization iterations more.

# In[52]:
plot_conv_weights(weights=weights_conv1)

# ## Close TensorFlow Session
session.close()
print_time_usage(start_time)

# ## Conclusion
# This tutorial showed how to save and retrieve the variables of a neural network in TensorFlow. This can be used in different ways. For example, if you want to use a neural network for recognizing images then you only have to train the network once and you can then deploy the finished network on other computers.
# Another use of checkpoints is if you have a very large neural network and data-set, then you may want to save checkpoints at regular intervals in case the computer crashes, so you can continue the optimization at a recent checkpoint instead of having to restart the optimization from the beginning.
# This tutorial also showed how to use the validation-set for so-called Early Stopping, where the optimization was aborted if it did not regularly improve the validation error. This is useful if the neural network starts to overfit and learn the noise of the training-set; although it was not really an issue with the convolutional network and MNIST data-set used in this tutorial.
# An interesting observation was that the convolutional weights (or filters) changed very little from the optimization, even though the performance of the network went from random guesses to near-perfect classification. It seems strange that the random weights were almost good enough. Why do you think this happens?

# ## Exercises
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# You may want to backup this Notebook before making any changes.
# * Optimization is stopped after 1000 iterations without improvement. Is this enough? Can you think of a better way to do Early Stopping? Try and implement it.
# * If the checkpoint file already exists then load it instead of doing the optimization.
# * Save a new checkpoint for every 100 optimization iterations. Retrieve the latest using `saver.latest_checkpoint()`. Why would you want to save multiple checkpionts instead of just the most recent?
# * Try and change the neural network, e.g. by adding another layer. What happens when you reload the variables from a different network?
# * Plot the weights for the 2nd convolutional layer before and after optimization using the function `plot_conv_weights()`. Are they almost identical as well?
# * Why do you think the optimized convolutional weights are almost the same as the random initialization?
# * Remake the program yourself without looking too much at this source-code.
# * Explain to a friend how the program works.

