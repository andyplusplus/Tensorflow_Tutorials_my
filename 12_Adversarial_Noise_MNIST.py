# # TensorFlow Tutorial #12

# # Adversarial Noise for MNIST

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
import time
from datetime import timedelta
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

# ### Helper-function for plotting images  # In[6]:
def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9
    
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        image = images[i].reshape(img_shape)
        
        image += noise
        
        image = np.clip(image, 0.0, 1.0)
        ax.imshow(image,
                  cmap='binary', interpolation='nearest')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    if is_plot: plt.show()

# ### Plot a few images to see if data is correct  # In[7]:
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
plot_images(images=images, cls_true=cls_true)

# ## TensorFlow Graph

# ### Placeholder variables  # In[8]:
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')  # In[9]:
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])  # In[10]:
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')  # In[11]:
y_true_cls = tf.argmax(y_true, axis=1)

# ### Adversarial Noise  # In[12]:
noise_limit = 0.35  # In[13]:
noise_l2_weight = 0.02  # In[14]:
ADVERSARY_VARIABLES = 'adversary_variables'  # In[15]:
collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]  # In[16]:
x_noise = tf.Variable(tf.zeros([img_size, img_size, num_channels]),
                      name='x_noise', trainable=False,
                      collections=collections)  # In[17]:
x_noise_clip = tf.assign(x_noise, tf.clip_by_value(x_noise,
                                                   -noise_limit,
                                                   noise_limit))  # In[18]:
x_noisy_image = x_image + x_noise  # In[19]:
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

# ### Convolutional Neural Network  # In[20]:
net = x_noisy_image
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
net = tf.contrib.layers.flatten(net)
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)
logits = net
y_pred = tf.nn.softmax(logits=logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                           logits=logits)
loss = tf.reduce_mean(cross_entropy)

# ### Optimizer for Normal Training  # In[21]:
[var.name for var in tf.trainable_variables()]  # In[22]:
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# ### Optimizer for Adversarial Noise  # In[23]:
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)  # In[24]:
[var.name for var in adversary_variables]  # In[25]:
l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(x_noise)  # In[26]:
loss_adversary = loss + l2_loss_noise  # In[27]:
optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables)

# ### Performance Measures  # In[28]:
y_pred_cls = tf.argmax(y_pred, axis=1)  # In[29]:
correct_prediction = tf.equal(y_pred_cls, y_true_cls)  # In[30]:
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ## TensorFlow Run

# ### Create TensorFlow session  # In[31]:
session = tf.Session()

# ### Initialize variables  # In[32]:
session.run(tf.global_variables_initializer())  # In[33]:
def init_noise():
    session.run(tf.variables_initializer([x_noise]))  # In[34]:
init_noise()

# ### Helper-function to perform optimization iterations  # In[35]:
train_batch_size = 64  # In[36]:
def optimize(num_iterations, adversary_target_cls=None):
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)
        if adversary_target_cls is not None:
            
            y_true_batch = np.zeros_like(y_true_batch)
            y_true_batch[:, adversary_target_cls] = 1.0
            
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        if adversary_target_cls is None:
            session.run(optimizer, feed_dict=feed_dict_train)
        else:
            session.run(optimizer_adversary, feed_dict=feed_dict_train)
            
            session.run(x_noise_clip)
        if (i % 100 == 0) or (i == num_iterations - 1):
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# ### Helper-functions for getting and plotting the noise  # In[37]:
def get_noise():
    noise = session.run(x_noise)
    return np.squeeze(noise)  # In[38]:
def plot_noise():
    noise = get_noise()
    
    print("Noise:")
    print("- Min:", noise.min())
    print("- Max:", noise.max())
    print("- Std:", noise.std())
    plt.imshow(noise, interpolation='nearest', cmap='seismic',
               vmin=-1.0, vmax=1.0)

# ### Helper-function to plot example errors  # In[39]:
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    
    images = data.x_test[incorrect]
    
    cls_pred = cls_pred[incorrect]
    cls_true = data.y_test_cls[incorrect]
    noise = get_noise()
    
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                noise=noise)

# ### Helper-function to plot confusion matrix  # In[40]:
def plot_confusion_matrix(cls_pred):
    cls_true = data.y_test_cls
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)

# ### Helper-function for showing the performance  # In[41]:
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

# ## Normal optimization of neural network  # In[42]:
optimize(num_iterations=1000)  # In[43]:
print_test_accuracy(show_example_errors=True)

# ## Find the adversarial noise  # In[44]:
init_noise()  # In[45]:
optimize(num_iterations=1000, adversary_target_cls=3)  # In[46]:
plot_noise()  # In[47]:
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ## Adversarial noise for all target-classes  # In[48]:
def find_all_noise(num_iterations=1000):
    all_noise = []
    for i in range(num_classes):
        print("Finding adversarial noise for target-class:", i)
        init_noise()
        optimize(num_iterations=num_iterations,
                 adversary_target_cls=i)
        noise = get_noise()
        all_noise.append(noise)
        print()
    
    return all_noise  # In[49]:
all_noise = find_all_noise(num_iterations=300)

# ### Plot the adversarial noise for all target-classes  # In[50]:
def plot_all_noise(all_noise):    
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        noise = all_noise[i]
        
        ax.imshow(noise,
                  cmap='seismic', interpolation='nearest',
                  vmin=-1.0, vmax=1.0)
        ax.set_xlabel(i)
        ax.set_xticks([])
        ax.set_yticks([])
    
    if is_plot: plt.show()  # In[51]:
plot_all_noise(all_noise)

# ## Immunity to adversarial noise

# ### Helper-function to make a neural network immune to noise  # In[52]:
def make_immune(target_cls, num_iterations_adversary=500,
                num_iterations_immune=200):
    print("Target-class:", target_cls)
    print("Finding adversarial noise ...")
    optimize(num_iterations=num_iterations_adversary,
             adversary_target_cls=target_cls)
    print()
    
    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False)
    print()
    print("Making the neural network immune to the noise ...")
    optimize(num_iterations=num_iterations_immune)
    print()
    
    print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False)

# ### Make immune to noise for target-class 3  # In[53]:
make_immune(target_cls=3)  # In[54]:
make_immune(target_cls=3)

# ### Make immune to noise for all target-classes  # In[55]:
for i in range(10):
    make_immune(target_cls=i)
    
    print()

# ### Make immune to all target-classes (double runs)  # In[56]:
for i in range(10):
    make_immune(target_cls=i)
    
    print()
    
    make_immune(target_cls=i)
    print()

# ### Plot the adversarial noise  # In[57]:
plot_noise()  # In[58]:
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ### Performance on clean images  # In[59]:
init_noise()  # In[60]:
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ## Close TensorFlow Session  # In[61]:
print_time_usage(start_time_global)

# ## Discussion

# ## Conclusion

# ## Exercises

# ## License (MIT)
