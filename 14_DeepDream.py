# # TensorFlow Tutorial #14

# # DeepDream

# ## Introduction

# ## Flowchart  # In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ### Recursive Optimization  # In[2]:

# ## Imports  # In[3]:
import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np
import random
import math
import PIL.Image
from scipy.ndimage.filters import gaussian_filter  # In[4]:
tf.__version__

# ## Inception Model  # In[5]:
import inception5h  # In[6]:  # In[7]:
inception5h.maybe_download()  # In[8]:
model = inception5h.Inception5h()  # In[9]:
len(model.layer_tensors)

# ## Helper-functions for image manipulation  # In[10]:
def load_image(filename):
    image = PIL.Image.open(filename)
    return np.float32(image)  # In[11]:
def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)

    image = image.astype(np.uint8)

    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')  # In[12]:
def plot_image(image):
    
    if False:
        image = np.clip(image/255.0, 0.0, 1.0)

        plt.imshow(image, interpolation='lanczos')
        if is_plot: plt.show()
    else:
        image = np.clip(image, 0.0, 255.0)

        image = image.astype(np.uint8)
        display(PIL.Image.fromarray(image))  # In[13]:
def normalize_image(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm  # In[14]:
def plot_gradient(gradient):
    gradient_normalized = normalize_image(gradient)

    plt.imshow(gradient_normalized, interpolation='bilinear')
    if is_plot: plt.show()  # In[15]:
def resize_image(image, size=None, factor=None):
    if factor is not None:
        size = np.array(image.shape[0:2]) * factor

        size = size.astype(int)
    else:
        size = size[0:2]

    size = tuple(reversed(size))
    img = np.clip(image, 0.0, 255.0)

    img = img.astype(np.uint8)

    img = PIL.Image.fromarray(img)

    img_resized = img.resize(size, PIL.Image.LANCZOS)

    img_resized = np.float32(img_resized)
    return img_resized

# ## DeepDream Algorithm

# ### Gradient  # In[16]:
def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """
    num_tiles = int(round(num_pixels / tile_size))

    num_tiles = max(1, num_tiles)

    actual_tile_size = math.ceil(num_pixels / num_tiles)
    return actual_tile_size  # In[17]:
def tiled_gradient(gradient, image, tile_size=400):
    grad = np.zeros_like(image)
    x_max, y_max, _ = image.shape
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    x_tile_size4 = x_tile_size // 4
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    y_tile_size4 = y_tile_size // 4
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)
    while x_start < x_max:
        x_end = x_start + x_tile_size

        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)
        while y_start < y_max:
            y_end = y_start + y_tile_size
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)
            img_tile = image[x_start_lim:x_end_lim,
                             y_start_lim:y_end_lim, :]
            feed_dict = model.create_feed_dict(image=img_tile)
            g = session.run(gradient, feed_dict=feed_dict)
            g /= (np.std(g) + 1e-8)
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g

            y_start = y_end
        x_start = x_end
    return grad

# ### Optimize Image  # In[18]:
def optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):
    """
    Use gradient ascent to optimize an image so it maximizes the
    mean value of the given layer_tensor.
    Parameters:
    layer_tensor: Reference to a tensor that will be maximized.
    image: Input image used as the starting point.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    show_gradient: Plot the gradient in each iteration.
    """
    img = image.copy()
    print("Image before:")
    plot_image(img)
    print("Processing image: ", end="")
    gradient = model.get_gradient(layer_tensor)
    for i in range(num_iterations):
        grad = tiled_gradient(gradient=gradient, image=img)

        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
        step_size_scaled = step_size / (np.std(grad) + 1e-8)
        img += grad * step_size_scaled
        if show_gradient:
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))
            plot_gradient(grad)
        else:
            print(". ", end="")
    print()
    print("Image after:")
    plot_image(img)
    return img

# ### Recursive Image Optimization  # In[19]:
def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):
    """
    Recursively blur and downscale the input image.
    Each downscaled image is run through the optimize_image()
    function to amplify the patterns that the Inception model sees.
    Parameters:
    image: Input image used as the starting point.
    rescale_factor: Downscaling factor for the image.
    num_repeats: Number of times to downscale the image.
    blend: Factor for blending the original and processed images.
    Parameters passed to optimize_image():
    layer_tensor: Reference to a tensor that will be maximized.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    """
    if num_repeats>0:
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)

        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats-1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)

        img_upscaled = resize_image(image=img_result, size=image.shape)
        image = blend * image + (1.0 - blend) * img_upscaled
    print("Recursive level:", num_repeats)
    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)
    return img_result

# ## TensorFlow Session  # In[20]:
session = tf.InteractiveSession(graph=model.graph)

# ## Hulk  # In[21]:
image = load_image(filename='images/hulk.jpg')
plot_image(image)  # In[22]:
layer_tensor = model.layer_tensors[2]
layer_tensor  # In[23]:
img_result = optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=6.0, tile_size=400,
                   show_gradient=True)  # In[24]:  # In[25]:
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)  # In[ ]:
layer_tensor = model.layer_tensors[6]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)  # In[ ]:
layer_tensor = model.layer_tensors[7][:,:,:,0:3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)  # In[ ]:
layer_tensor = model.layer_tensors[11][:,:,:,0]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)

# ## Giger  # In[ ]:
image = load_image(filename='images/giger.jpg')
plot_image(image)  # In[ ]:
layer_tensor = model.layer_tensors[3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)  # In[ ]:
layer_tensor = model.layer_tensors[5]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)

# ## Escher  # In[ ]:
image = load_image(filename='images/escher_planefilling2.jpg')
plot_image(image)  # In[ ]:
layer_tensor = model.layer_tensors[6]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)

# ## Close TensorFlow Session  # In[ ]:
print_time_usage(start_time_global)

# ## Conclusion

# ## Exercises

# ## License (MIT)
