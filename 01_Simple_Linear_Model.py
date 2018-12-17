# # TensorFlow Tutorial #01
# # Simple Linear Model

# In[1]: # ## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
is_plot = False
import numpy as np
from sklearn.metrics import confusion_matrix


# In[3]: # ## Load Data # The MNIST data-set is about 12 MB and will be downloaded automatically if it is not located in the given path.
from common.mnist_common import get_mnist
data = get_mnist()

img_size = data.img_size # The number of pixels in each dimension of an image.
img_size_flat = data.img_size_flat # The images are stored in one-dimensional arrays of this length.
img_shape = data.img_shape # Tuple with height and width of images used to reshape arrays.
num_classes = data.num_classes # Number of classes, one class for each of 10 digits.
num_channels = data.num_channels # Number of colour channels for the images: 1 channel for gray-scale.

# In[9]: # ### Plot a few images to see if data is correct
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
from common.plot_helper import plot_images # In[8]: # ### Helper-function for plotting images # Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
plot_images(images=images, cls_true=cls_true)




# ## TensorFlow Graph    # ### Placeholder variables
# In[10]: variables for input
x = tf.placeholder(tf.float32, [None, img_size_flat])    # input image
y_true = tf.placeholder(tf.float32, [None, num_classes]) # onehot array
y_true_cls = tf.placeholder(tf.int64, [None])            # argmax


# In[13]: # ### Variables to be optimized
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))


# In[15]: # ### Model
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)


# In[18]: # ### Cost-function to be optimized   TOBEHERE
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)


# In[20]: # ### Optimization method
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)


# In[21]: # ### Performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# ## TensorFlow Run
# In[23]: # ### Create TensorFlow session
session = tf.Session()
session.run(tf.global_variables_initializer())

# In[25]: # ### Helper-function to perform optimization iterations # In[26]: # Function for performing a number of optimization iterations so as to gradually improve the `weights` and `biases` of the model. In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples.
def optimize(num_iterations):
    for i in range(num_iterations):
        batch_size = 100
        x_batch, y_true_batch, _ = data.random_batch(batch_size=batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)


# In[27]: # ### Helper-functions to show performance
feed_dict_test = {x: data.x_test,
                  y_true: data.y_test,
                  y_true_cls: data.y_test_cls}


def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))


# In[29]: # Function for printing and plotting the confusion matrix using scikit-learn.
def print_confusion_matrix():
    cls_true = data.y_test_cls
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.tight_layout()

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if is_plot: plt.show()

# In[30]: # Function for plotting examples of images from the test-set that have been mis-classified.
def plot_example_errors():
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)
    incorrect = (correct == False)

    images = data.x_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.y_test_cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# In[31]: # ### Helper-function to plot the model weights # Function for plotting the `weights` of the model. 10 images are plotted, one for each digit that the model is trained to recognize.
def plot_weights():
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i<10:
            image = w[:, i].reshape(img_shape)
            ax.set_xlabel("Weights: {0}".format(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    if is_plot: plt.show()

# In[32]: # ## Performance before any optimization # The accuracy on the test-set is 9.8%. This is because the model has only been initialized and not optimized at all, so it always predicts that the image shows a zero digit, as demonstrated in the plot below, and it turns out that 9.8% of the images in the test-set happens to be zero digits.
print_accuracy()
plot_example_errors()

# In[34]: # ## Performance after 1 optimization iteration # Already after a single optimization iteration, the model has increased its accuracy on the test-set significantly.
optimize(num_iterations=1)
print_accuracy()
plot_example_errors()
plot_weights()

# In[38]: # ## Performance after 10 optimization iterations # We have already performed 1 iteration.
optimize(num_iterations=9)
print_accuracy()
plot_example_errors()
plot_weights()

# In[42]: # ## Performance after 1000 optimization iterations # We have already performed 10 iterations.
optimize(num_iterations=990)
print_accuracy()
plot_example_errors()
plot_weights()

# In[46]: # We can also print and plot the so-called confusion matrix which lets us see more details about the mis-classifications. For example, it shows that images actually depicting a 5 have sometimes been mis-classified as all other possible digits, but mostly as 6 or 8.
print_confusion_matrix()

# In[47]: # This has been commented out in case you want to modify and experiment # with the Notebook without having to restart it.
session.close()

# ## Exercises
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# You may want to backup this Notebook before making any changes.
# * Change the learning-rate for the optimizer.
# * Change the optimizer to e.g. `AdagradOptimizer` or `AdamOptimizer`. TOBEHERE
# * Change the batch-size to e.g. 1 or 1000.
# * How do these changes affect the performance?
# * Do you think these changes will have the same effect (if any) on other classification problems and mathematical models?
# * Do you get the exact same results if you run the Notebook multiple times without changing any parameters? Why or why not?
# * Change the function `plot_example_errors()` so it also prints the `logits` and `y_pred` values for the mis-classified examples.
# * Use `sparse_softmax_cross_entropy_with_logits` instead of `softmax_cross_entropy_with_logits`. This may require several changes to multiple places in the source-code. Discuss the advantages and disadvantages of using the two methods.
# * Remake the program yourself without looking too much at this source-code.
# * Explain to a friend how the program works.