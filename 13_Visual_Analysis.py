# # TensorFlow Tutorial #13

# # Visual Analysis

# ## Introduction

# ## Flowchart  # In[44]:

# ## Imports  # In[2]:
import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np
import inception  # In[3]:
tf.__version__

# ## Inception Model

# ### Download the Inception model from the internet  # In[4]:  # In[5]:
inception.maybe_download()

# ### Names of convolutional layers  # In[6]:
def get_conv_layer_names():
    model = inception.Inception()

    names = [op.name for op in model.graph.get_operations() if op.type=='Conv2D']
    model.close()
    return names  # In[7]:
conv_names = get_conv_layer_names()  # In[8]:
len(conv_names)  # In[9]:
conv_names[:5]  # In[10]:
conv_names[-5:]

# ## Helper-function for finding the input image  # In[11]:
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
    model = inception.Inception()
    resized_image = model.resized_image
    y_pred = model.y_pred
    if conv_id is None:
        loss = model.y_logits[0, feature]
    else:
        conv_name = conv_names[conv_id]

        tensor = model.graph.get_tensor_by_name(conv_name + ":0")
        with model.graph.as_default():
            loss = tf.reduce_mean(tensor[:,:,:,feature])

    gradient = tf.gradients(loss, resized_image)
    session = tf.Session(graph=model.graph)
    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0
    for i in range(num_iterations):
        feed_dict = {model.tensor_name_resized_image: image}
        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        grad = np.array(grad).squeeze()
        step_size = 1.0 / (grad.std() + 1e-8)
        image += step_size * grad
        image = np.clip(image, 0.0, 255.0)
        if show_progress:
            print("Iteration:", i)
            pred = np.squeeze(pred)
            pred_cls = np.argmax(pred)
            cls_name = model.name_lookup.cls_to_name(pred_cls,
                                               only_first_name=True)
            cls_score = pred[pred_cls]
            msg = "Predicted class-name: {0} (#{1}), score: {2:>7.2%}"
            print(msg.format(cls_name, pred_cls, cls_score))
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))
            print("Loss:", loss_value)
            print()
    model.close()
    return image.squeeze()

# ### Helper-function for plotting image and noise  # In[12]:
def normalize_image(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm  # In[13]:
def plot_image(image):
    img_norm = normalize_image(image)

    plt.imshow(img_norm, interpolation='nearest')
    if is_plot: plt.show()  # In[14]:
def plot_images(images, show_size=100):
    """
    The show_size is the number of pixels to show for each image.
    The max value is 299.
    """
    fig, axes = plt.subplots(2, 3)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    smooth = True

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    for i, ax in enumerate(axes.flat):
        img = images[i, 0:show_size, 0:show_size, :]

        img_norm = normalize_image(img)

        ax.imshow(img_norm, interpolation=interpolation)
        ax.set_xticks([])
        ax.set_yticks([])
    if is_plot: plt.show()    

# ### Helper-function for optimizing and plotting images  # In[15]:
def optimize_images(conv_id=None, num_iterations=30, show_size=100):
    """
    Find 6 images that maximize the 6 first features in the layer
    given by the conv_id.
    Parameters:
    conv_id: Integer identifying the convolutional layer to
             maximize. It is an index into conv_names.
             If None then use the last layer before the softmax output.
    num_iterations: Number of optimization iterations to perform.
    show_size: Number of pixels to show for each image. Max 299.
    """
    if conv_id is None:
        print("Final fully-connected layer before softmax.")
    else:
        print("Layer:", conv_names[conv_id])
    images = []
    for feature in range(1,7):
        print("Optimizing image for feature no.", feature)

        image = optimize_image(conv_id=conv_id, feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)
        image = image.squeeze()
        images.append(image)
    images = np.array(images)
    plot_images(images=images, show_size=show_size)

# ## Results

# ### Optimize a single image for an early convolutional layer  # In[16]:
image = optimize_image(conv_id=5, feature=2,
                       num_iterations=30, show_progress=True)  # In[17]:
plot_image(image)

# ### Optimize multiple images for convolutional layers  # In[18]:
optimize_images(conv_id=0, num_iterations=10)  # In[19]:
optimize_images(conv_id=3, num_iterations=30)  # In[20]:
optimize_images(conv_id=4, num_iterations=30)  # In[21]:
optimize_images(conv_id=5, num_iterations=30)  # In[22]:
optimize_images(conv_id=6, num_iterations=30)  # In[23]:
optimize_images(conv_id=7, num_iterations=30)  # In[24]:
optimize_images(conv_id=8, num_iterations=30)  # In[25]:
optimize_images(conv_id=9, num_iterations=30)  # In[26]:
optimize_images(conv_id=10, num_iterations=30)  # In[27]:
optimize_images(conv_id=20, num_iterations=30)  # In[28]:
optimize_images(conv_id=30, num_iterations=30)  # In[29]:
optimize_images(conv_id=40, num_iterations=30)  # In[30]:
optimize_images(conv_id=50, num_iterations=30)  # In[31]:
optimize_images(conv_id=60, num_iterations=30)  # In[32]:
optimize_images(conv_id=70, num_iterations=30)  # In[33]:
optimize_images(conv_id=80, num_iterations=30)  # In[34]:
optimize_images(conv_id=90, num_iterations=30)  # In[35]:
optimize_images(conv_id=93, num_iterations=30)

# ### Final fully-connected layer before Softmax  # In[36]:
optimize_images(conv_id=None, num_iterations=30)  # In[37]:
image = optimize_image(conv_id=None, feature=1,
                       num_iterations=100, show_progress=True)  # In[38]:
plot_image(image=image)

# ## Close TensorFlow Session

# ## Conclusion

# ### Other Methods

# ## Exercises

# ## License (MIT)
