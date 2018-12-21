# # TensorFlow Tutorial #09

# # Video Data

# ## WARNING!

# ## Introduction

# ## Flowchart  # In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ## Imports  # In[2]:
import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np
import time
from datetime import timedelta
import os
import inception
import prettytensor as pt  # In[3]:
tf.__version__  # In[4]:
pt.__version__

# ## Load Data  # In[5]:
import knifey  # In[6]:
from knifey import num_classes  # In[7]:  # In[8]:
data_dir = knifey.data_dir  # In[9]:
knifey.maybe_download_and_extract()  # In[10]:
dataset = knifey.load()

# ### Your Data  # In[11]:

# ### Training and Test-Sets  # In[12]:
class_names = dataset.class_names
class_names  # In[13]:
image_paths_train, cls_train, labels_train = dataset.get_training_set()  # In[14]:
image_paths_train[0]  # In[15]:
image_paths_test, cls_test, labels_test = dataset.get_test_set()  # In[16]:
image_paths_test[0]  # In[17]:
print("Size of:")
print("- Training-set:\t\t{}".format(len(image_paths_train)))
print("- Test-set:\t\t{}".format(len(image_paths_test)))

# ### Helper-function for plotting images  # In[18]:
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(3, 3)
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i],
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

    if is_plot: plt.show()

# ### Helper-function for loading images  # In[19]:
from matplotlib.image import imread
def load_images(image_paths):
    images = [imread(path) for path in image_paths]
    return np.asarray(images)

# ### Plot a few images to see if data is correct  # In[20]:
images = load_images(image_paths=image_paths_test[0:9])
cls_true = cls_test[0:9]
plot_images(images=images, cls_true=cls_true, smooth=True)

# ## Download the Inception Model  # In[21]:  # In[22]:
inception.maybe_download()

# ## Load the Inception Model  # In[23]:
model = inception.Inception()

# ## Calculate Transfer-Values  # In[24]:
from inception import transfer_values_cache  # In[25]:
file_path_cache_train = os.path.join(data_dir, 'inception-knifey-train.pkl')
file_path_cache_test = os.path.join(data_dir, 'inception-knifey-test.pkl')  # In[26]:
print("Processing Inception transfer-values for training-images ...")
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              image_paths=image_paths_train,
                                              model=model)  # In[27]:
print("Processing Inception transfer-values for test-images ...")
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             image_paths=image_paths_test,
                                             model=model)  # In[28]:
transfer_values_train.shape  # In[29]:
transfer_values_test.shape

# ### Helper-function for plotting transfer-values  # In[30]:
def plot_transfer_values(i):
    print("Input image:")

    image = imread(image_paths_test[i])
    plt.imshow(image, interpolation='spline16')
    if is_plot: plt.show()
    print("Transfer-values for the image using Inception model:")

    img = transfer_values_test[i]
    img = img.reshape((32, 64))
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    if is_plot: plt.show()  # In[31]:
plot_transfer_values(i=100)  # In[32]:
plot_transfer_values(i=300)

# ## Analysis of Transfer-Values using PCA  # In[33]:
from sklearn.decomposition import PCA  # In[34]:
pca = PCA(n_components=2)  # In[35]:
transfer_values = transfer_values_train  # In[36]:
cls = cls_train  # In[37]:
transfer_values.shape  # In[38]:
transfer_values_reduced = pca.fit_transform(transfer_values)  # In[39]:
transfer_values_reduced.shape  # In[40]:
def plot_scatter(values, cls):
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
    idx = np.random.permutation(len(values))

    colors = cmap[cls[idx]]
    x = values[idx, 0]
    y = values[idx, 1]
    plt.scatter(x, y, color=colors, alpha=0.5)
    if is_plot: plt.show()  # In[41]:
plot_scatter(transfer_values_reduced, cls=cls)

# ## Analysis of Transfer-Values using t-SNE  # In[42]:
from sklearn.manifold import TSNE  # In[43]:
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)  # In[44]:
tsne = TSNE(n_components=2)  # In[45]:
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)   # In[46]:
transfer_values_reduced.shape  # In[47]:
plot_scatter(transfer_values_reduced, cls=cls)

# ## New Classifier in TensorFlow

# ### Placeholder Variables  # In[48]:
transfer_len = model.transfer_len  # In[49]:
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')  # In[50]:
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')  # In[51]:
y_true_cls = tf.argmax(y_true, dimension=1)

# ### Neural Network  # In[52]:
x_pretty = pt.wrap(x)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.        fully_connected(size=1024, name='layer_fc1').        softmax_classifier(num_classes=num_classes, labels=y_true)

# ### Optimization Method  # In[53]:
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)  # In[54]:
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

# ### Classification Accuracy  # In[55]:
y_pred_cls = tf.argmax(y_pred, dimension=1)  # In[56]:
correct_prediction = tf.equal(y_pred_cls, y_true_cls)  # In[57]:
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ## TensorFlow Run

# ### Create TensorFlow Session  # In[58]:
session = tf.Session()

# ### Initialize Variables  # In[59]:
session.run(tf.global_variables_initializer())

# ### Helper-function to get a random training-batch  # In[60]:
train_batch_size = 64  # In[61]:
def random_batch():
    num_images = len(transfer_values_train)
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]
    return x_batch, y_batch

# ### Helper-function to perform optimization  # In[62]:
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
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# ## Helper-Functions for Showing Results

# ### Helper-function to plot example errors  # In[63]:
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    idx = np.flatnonzero(incorrect)
    n = min(len(idx), 9)

    idx = np.random.choice(idx,
                           size=n,
                           replace=False)
    cls_pred = cls_pred[idx]
    cls_true = cls_test[idx]
    image_paths = [image_paths_test[i] for i in idx]
    images = load_images(image_paths)
    plot_images(images=images,
                cls_true=cls_true,
                cls_pred=cls_pred)

# ### Helper-function to plot confusion matrix  # In[64]:
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.
    for i in range(num_classes):
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

# ### Helper-functions for calculating classifications  # In[65]:
batch_size = 256
def predict_cls(transfer_values, labels, cls_true):
    num_images = len(transfer_values)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0
    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    correct = (cls_true == cls_pred)
    return correct, cls_pred  # In[66]:
def predict_cls_test():
    return predict_cls(transfer_values = transfer_values_test,
                       labels = labels_test,
                       cls_true = cls_test)

# ### Helper-functions for calculating the classification accuracy  # In[67]:
def classification_accuracy(correct):
    return correct.mean(), correct.sum()

# ### Helper-function for showing the classification accuracy  # In[68]:
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
        plot_confusion_matrix(cls_pred=cls_pred)

# ## Results

# ## Performance before any optimization  # In[69]:
print_test_accuracy(show_example_errors=False,
                    show_confusion_matrix=True)

# ## Performance after 1000 optimization iterations  # In[70]:
optimize(num_iterations=1000)  # In[71]:
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# ## Close TensorFlow Session  # In[72]:
print_time_usage(start_time_global)

# ## Conclusion

# ## Exercises
