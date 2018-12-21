# # TensorFlow Tutorial #13-B

# # Visual Analysis (MNIST)

# ## Introduction

# ## Flowchart

# ## Imports  # In[1]:
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
import math  # In[2]:
tf.__version__

# ## Load Data  # In[3]:
from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")  # In[4]:
print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))  # In[5]:
img_size = data.img_size
img_size_flat = data.img_size_flat
img_shape = data.img_shape
num_classes = data.num_classes
num_channels = data.num_channels

# ### Helper-functions for plotting images  # In[6]:
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

    if is_plot: plt.show()  # In[7]:
def plot_images10(images, smooth=True):
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        img = images[i, :, :]

        ax.imshow(img, interpolation=interpolation, cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    if is_plot: plt.show()      # In[8]:
def plot_image(image):
    plt.imshow(image, interpolation='nearest', cmap='binary')
    plt.xticks([])
    plt.yticks([])

# ### Plot a few images to see if data is correct  # In[9]:
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
plot_images(images=images, cls_true=cls_true)

# ## TensorFlow Graph

# ### Placeholder variables  # In[10]:
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')  # In[11]:
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])  # In[12]:
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')  # In[13]:
y_true_cls = tf.argmax(y_true, axis=1)

# ### Neural Network  # In[14]:
net = x_image  # In[15]:
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)  # In[16]:
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)  # In[17]:
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)  # In[18]:
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)  # In[19]:
net = tf.contrib.layers.flatten(net)  # In[20]:
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)  # In[21]:
net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)  # In[22]:
logits = net  # In[23]:
y_pred = tf.nn.softmax(logits=logits)  # In[24]:
y_pred_cls = tf.argmax(y_pred, axis=1)

# ### Loss-Function to be Optimized  # In[25]:
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)  # In[26]:
loss = tf.reduce_mean(cross_entropy)

# ### Optimization Method  # In[27]:
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# ### Classification Accuracy  # In[28]:
correct_prediction = tf.equal(y_pred_cls, y_true_cls)  # In[29]:
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ## Optimize the Neural Network

# ### Create TensorFlow session  # In[30]:
session = tf.Session()

# ### Initialize variables  # In[31]:
session.run(tf.global_variables_initializer())

# ### Helper-function to perform optimization iterations  # In[32]:
train_batch_size = 64  # In[33]:
total_iterations = 0
def optimize(num_iterations):
    global total_iterations
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
    total_iterations += num_iterations

# ### Helper-function to plot example errors  # In[34]:
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)

    images = data.x_test[incorrect]

    cls_pred = cls_pred[incorrect]
    cls_true = data.y_test_cls[incorrect]

    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# ### Helper-function to plot confusion matrix  # In[35]:
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

# ### Helper-function for showing the performance  # In[36]:
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
        feed_dict = {x: images,
                     y_true: labels}
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

# ### Performance before any optimization  # In[37]:
print_test_accuracy()

# ### Performance after 10,000 optimization iterations  # In[38]:  # In[39]:
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ## Optimizing the Input Images

# ### Helper-function for getting the names of convolutional layers  # In[40]:
def get_conv_layer_names():
    graph = tf.get_default_graph()

    names = [op.name for op in graph.get_operations() if op.type=='Conv2D']
    return names  # In[41]:
conv_names = get_conv_layer_names()  # In[42]:
conv_names  # In[43]:
len(conv_names)

# ### Helper-function for finding the input image  # In[44]:
def optimize_image(conv_id=None, feature=0,
                   num_iterations=30, show_progress=True):
    """
    Find an image that maximizes the feature
    given by the conv_id and feature number.
    Parameters:
    conv_id: Integer identifying the convolutional layer to
             maximize. It is an index into conv_names.
             If None then use the last fully-connected layer
             before the softmax output.
    feature: Index into the layer for the feature to maximize.
    num_iteration: Number of optimization iterations to perform.
    show_progress: Boolean whether to show the progress.
    """
    if conv_id is None:
        loss = tf.reduce_mean(logits[:, feature])
    else:
        conv_name = conv_names[conv_id]

        graph = tf.get_default_graph()

        tensor = graph.get_tensor_by_name(conv_name + ":0")
        loss = tf.reduce_mean(tensor[:,:,:,feature])
    gradient = tf.gradients(loss, x_image)
    image = 0.1 * np.random.uniform(size=img_shape) + 0.45
    for i in range(num_iterations):
        img_reshaped = image[np.newaxis,:,:,np.newaxis]
        feed_dict = {x_image: img_reshaped}
        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        grad = np.array(grad).squeeze()
        step_size = 1.0 / (grad.std() + 1e-8)
        image += step_size * grad
        image = np.clip(image, 0.0, 1.0)
        if show_progress:
            print("Iteration:", i)
            pred = np.squeeze(pred)
            pred_cls = np.argmax(pred)
            cls_score = pred[pred_cls]
            msg = "Predicted class: {0}, score: {1:>7.2%}"
            print(msg.format(pred_cls, cls_score))
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))
            print("Loss:", loss_value)
            print()
    return image.squeeze()  # In[45]:
def optimize_images(conv_id=None, num_iterations=30):
    """
    Find 10 images that maximize the 10 first features in the layer
    given by the conv_id.
    Parameters:
    conv_id: Integer identifying the convolutional layer to
             maximize. It is an index into conv_names.
             If None then use the last layer before the softmax output.
    num_iterations: Number of optimization iterations to perform.
    """
    if conv_id is None:
        print("Final fully-connected layer before softmax.")
    else:
        print("Layer:", conv_names[conv_id])
    images = []
    for feature in range(0,10):
        print("Optimizing image for feature no.", feature)

        image = optimize_image(conv_id=conv_id, feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)
        image = image.squeeze()
        images.append(image)
    images = np.array(images)
    plot_images10(images=images)

# ### First Convolutional Layer  # In[46]:
optimize_images(conv_id=0)

# ### Second Convolutional Layer  # In[47]:
optimize_images(conv_id=1)

# ### Final output layer  # In[48]:
image = optimize_image(conv_id=None, feature=2,
                       num_iterations=10, show_progress=True)  # In[49]:
plot_image(image)  # In[50]:
optimize_images(conv_id=None)

# ### Close TensorFlow Session  # In[51]:
print_time_usage(start_time_global)

# ## Conclusion

# ## Exercises

# ## License (MIT)
