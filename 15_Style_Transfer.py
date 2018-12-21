# # TensorFlow Tutorial #15

# # Style Transfer

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
import PIL.Image  # In[3]:
tf.__version__

# ## VGG-16 Model  # In[4]:
import vgg16  # In[5]:  # In[6]:
vgg16.maybe_download()

# ## Helper-functions for image manipulation  # In[7]:
def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)
    if max_size is not None:
        factor = max_size / np.max(image.size)

        size = np.array(image.size) * factor
        size = size.astype(int)
        image = image.resize(size, PIL.Image.LANCZOS)
    return np.float32(image)  # In[8]:
def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)

    image = image.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')  # In[9]:
def plot_image_big(image):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    display(PIL.Image.fromarray(image))  # In[10]:
def plot_images(content_image, style_image, mixed_image):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    smooth = True

    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    if is_plot: plt.show()

# ## Loss Functions  # In[11]:
def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))  # In[12]:
def create_content_loss(session, model, content_image, layer_ids):
    """
    Create the loss-function for the content-image.
    Parameters:
    session: An open TensorFlow session for running the model's graph.
    model: The model, e.g. an instance of the VGG16-class.
    content_image: Numpy float array with the content-image.
    layer_ids: List of integer id's for the layers to use in the model.
    """

    feed_dict = model.create_feed_dict(image=content_image)
    layers = model.get_layer_tensors(layer_ids)
    values = session.run(layers, feed_dict=feed_dict)
    with model.graph.as_default():
        layer_losses = []

        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(layer, value_const)
            layer_losses.append(loss)
        total_loss = tf.reduce_mean(layer_losses)
    return total_loss  # In[13]:
def gram_matrix(tensor):
    shape = tensor.get_shape()

    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])

    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram                              # In[14]:
def create_style_loss(session, model, style_image, layer_ids):
    """
    Create the loss-function for the style-image.
    Parameters:
    session: An open TensorFlow session for running the model's graph.
    model: The model, e.g. an instance of the VGG16-class.
    style_image: Numpy float array with the style-image.
    layer_ids: List of integer id's for the layers to use in the model.
    """
    feed_dict = model.create_feed_dict(image=style_image)
    layers = model.get_layer_tensors(layer_ids)
    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]
        values = session.run(gram_layers, feed_dict=feed_dict)
        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(gram_layer, value_const)
            layer_losses.append(loss)
        total_loss = tf.reduce_mean(layer_losses)
    return total_loss  # In[15]:
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) +            tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
    return loss

# ## Style-Transfer Algorithm  # In[16]:
def style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):
    """
    Use gradient descent to find an image that minimizes the
    loss-functions of the content-layers and style-layers. This
    should result in a mixed-image that resembles the contours
    of the content-image, and resembles the colours and textures
    of the style-image.
    Parameters:
    content_image: Numpy 3-dim float-array with the content-image.
    style_image: Numpy 3-dim float-array with the style-image.
    content_layer_ids: List of integers identifying the content-layers.
    style_layer_ids: List of integers identifying the style-layers.
    weight_content: Weight for the content-loss-function.
    weight_style: Weight for the style-loss-function.
    weight_denoise: Weight for the denoising-loss-function.
    num_iterations: Number of optimization iterations to perform.
    step_size: Step-size for the gradient in each iteration.
    """
    model = vgg16.VGG16()
    session = tf.InteractiveSession(graph=model.graph)
    print("Content layers:")
    print(model.get_layer_names(content_layer_ids))
    print()
    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()
    loss_content = create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids)
    loss_style = create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)    
    loss_denoise = create_denoise_loss(model)
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')
    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))
    loss_combined = weight_content * adj_content * loss_content +                     weight_style * adj_style * loss_style +                     weight_denoise * adj_denoise * loss_denoise
    gradient = tf.gradients(loss_combined, model.input)
    run_list = [gradient, update_adj_content, update_adj_style,                 update_adj_denoise]
    mixed_image = np.random.rand(*content_image.shape) + 128
    for i in range(num_iterations):
        feed_dict = model.create_feed_dict(image=mixed_image)
        grad, adj_content_val, adj_style_val, adj_denoise_val         = session.run(run_list, feed_dict=feed_dict)
        grad = np.squeeze(grad)
        step_size_scaled = step_size / (np.std(grad) + 1e-8)
        mixed_image -= grad * step_size_scaled
        mixed_image = np.clip(mixed_image, 0.0, 255.0)
        print(". ", end="")
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))
            plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)
    print()
    print("Final image:")
    plot_image_big(mixed_image)
    session.close()
    return mixed_image

# ## Example  # In[17]:
content_filename = 'images/willy_wonka_old.jpg'
content_image = load_image(content_filename, max_size=None)  # In[18]:
style_filename = 'images/style7.jpg'
style_image = load_image(style_filename, max_size=300)  # In[19]:
content_layer_ids = [4]  # In[20]:
style_layer_ids = list(range(13))
print_time_usage(start_time_global)  # In[21]:

# ## Conclusion

# ## Exercises

# ## License (MIT)
