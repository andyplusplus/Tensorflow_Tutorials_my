# # TensorFlow Tutorial #06

# # CIFAR-10
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
import os
import prettytensor as pt
import cifar10
cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()  # In[9]: # Load the training-set. This returns the images, the class-numbers as integers, and the class-numbers as One-Hot encoded arrays called labels.
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# ### Data Dimensions
from cifar10 import img_size, num_channels, num_classes
img_size_cropped = 24  # In[14]: # ### Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
        cls_true_name = class_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    if is_plot: plt.show()  # In[15]: # Get the first images from the test-set.
images = images_test[0:9]
cls_true = cls_test[0:9]
plot_images(images=images, cls_true=cls_true, smooth=False) # Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=True)  # In[17]: # ## TensorFlow Graph
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)  # In[20]: # ### Helper-function for creating Pre-Processing
def pre_process_image(image, training):
    if training:
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image  # In[21]: # The function above is called for each image in the input batch using the following function.
def pre_process(images, training):
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    return images  # In[22]: # In order to plot the distorted images, we create the pre-processing graph for TensorFlow, so we may execute it later.
distorted_images = pre_process(images=x, training=True)  # In[23]: # ### Helper-function for creating Main Processing
def main_network(images, training):
    x_pretty = pt.wrap(images)
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).            max_pool(kernel=2, stride=2).            conv2d(kernel=5, depth=64, name='layer_conv2').            max_pool(kernel=2, stride=2).            flatten().            fully_connected(size=256, name='layer_fc1').            fully_connected(size=128, name='layer_fc2').            softmax_classifier(num_classes=num_classes, labels=y_true)
    return y_pred, loss  # In[24]: # ### Helper-function for creating Neural Network
def create_network(training):
    with tf.variable_scope('network', reuse=not training):
        images = x
        images = pre_process(images=images, training=training)
        y_pred, loss = main_network(images=images, training=training)
    return y_pred, loss  # In[25]: # ### Create Neural Network for Training Phase
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)  # In[26]: # Create the neural network to be used for training. The `create_network()` function returns both `y_pred` and `loss`, but we only need the `loss`-function during training.
_, loss = create_network(training=True)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)  # In[28]: # ### Create Neural Network for Test Phase / Inference
y_pred, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # In[32]: # ### Saver
saver = tf.train.Saver()  # In[33]: # ### Getting the Weights
def get_weights_variable(layer_name):
    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')
    return variable
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')  # In[35]: # ### Getting the Layer Outputs
def get_layer_output(layer_name):
    tensor_name = "network/" + layer_name + "/Relu:0"
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
    return tensor
output_conv1 = get_layer_output(layer_name='layer_conv1')
output_conv2 = get_layer_output(layer_name='layer_conv2')  # In[37]: # ## TensorFlow Run
session = tf.Session()  # In[38]: # ### Restore or initialize variables
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'cifar10_cnn')
try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(session, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())  # In[42]: # ### Helper-function to get a random training-batch
train_batch_size = 64
def random_batch():
    num_images = len(images_train)
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]
    return x_batch, y_batch  # In[44]: # ### Helper-function to perform optimization
def optimize(num_iterations):
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)
            print("Saved checkpoint.")
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))  # In[45]: # ### Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = images_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = cls_test[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])  # In[46]: # ### Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.
    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))  # In[47]: # ### Helper-functions for calculating classifications
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
    return correct, cls_pred  # In[48]: # Calculate the predicted class for the test-set.
def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)  # In[49]: # ### Helper-functions for the classification accuracy
def classification_accuracy(correct):
    return correct.mean(), correct.sum()  # In[50]: # ### Helper-function for showing the performance
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    correct, cls_pred = predict_cls_test()
    acc, num_correct = classification_accuracy(correct)
    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)  # In[51]: # ### Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    if is_plot: plt.show()  # In[52]: # ### Helper-function for plotting the output of convolutional layers
def plot_layer_output(layer_output, image):
    feed_dict = {x: [image]}
    values = session.run(layer_output, feed_dict=feed_dict)
    values_min = np.min(values)
    values_max = np.max(values)
    num_images = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_images))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_images:
            img = values[0, :, :, i]
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    if is_plot: plt.show()  # In[53]: # ## Examples of distorted input images
def plot_distorted_image(image, cls_true):
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)
    feed_dict = {x: image_duplicates}
    result = session.run(distorted_images, feed_dict=feed_dict)
    plot_images(images=result, cls_true=np.repeat(cls_true, 9))  # In[54]: # Helper-function for getting an image and its class-number from the test-set.
def get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]  # In[55]: # Get an image and its true class from the test-set.
img, cls = get_test_image(16)  # In[56]:
plot_distorted_image(img, cls)

# ## Perform optimization  # In[57]:
if False:
    optimize(num_iterations=1000)

# ## Results  # In[58]:
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ### Convolutional Weights  # In[59]:
plot_conv_weights(weights=weights_conv1, input_channel=0)  # In[60]:
plot_conv_weights(weights=weights_conv2, input_channel=1)

# ### Output of convolutional layers  # In[61]:
def plot_image(image):
    fig, axes = plt.subplots(1, 2)
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')

    if is_plot: plt.show()  # In[62]:
img, cls = get_test_image(16)
plot_image(img)  # In[63]:
plot_layer_output(output_conv1, image=img)  # In[64]:
plot_layer_output(output_conv2, image=img)

# ### Predicted class-labels  # In[65]:
label_pred, cls_pred = session.run([y_pred, y_pred_cls],
                                   feed_dict={x: [img]})  # In[66]:
np.set_printoptions(precision=3, suppress=True)
print(label_pred[0])  # In[67]:

# ## License (MIT)

# ## License (MIT)
class_names[3]  # In[68]:
class_names[5]

# ## Close TensorFlow Session  # In[69]:
session.close()
print_time_usage(start_time_global)

# ## Conclusion

# ## Exercises
