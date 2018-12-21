# # TensorFlow Tutorial #11

# # Adversarial Examples  # In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np
import os
import inception  # In[3]:

# ## Inception Model

# ### Download the Inception model from the internet  # In[4]:  # In[5]:
inception.maybe_download()

# ### Load the Inception Model  # In[6]:
model = inception.Inception()

# ### Get Input and Output for the Inception Model  # In[7]:
resized_image = model.resized_image  # In[8]:
y_pred = model.y_pred  # In[9]:
y_logits = model.y_logits

# ### Hack the Inception Model  # In[10]:
with model.graph.as_default():
    pl_cls_target = tf.placeholder(dtype=tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=[pl_cls_target])
    gradient = tf.gradients(loss, resized_image)

# ## TensorFlow Session  # In[11]:
session = tf.Session(graph=model.graph)

# ## Helper-function for finding Adversary Noise  # In[12]:
def find_adversary_noise(image_path, cls_target, noise_limit=3.0,
                         required_score=0.99, max_iterations=100):
    """
    Find the noise that must be added to the given image so
    that it is classified as the target-class.
    image_path: File-path to the input-image (must be *.jpg).
    cls_target: Target class-number (integer between 1-1000).
    noise_limit: Limit for pixel-values in the noise.
    required_score: Stop when target-class score reaches this.
    max_iterations: Max number of optimization iterations to perform.
    """
    feed_dict = model._create_feed_dict(image_path=image_path)
    pred, image = session.run([y_pred, resized_image],
                              feed_dict=feed_dict)
    pred = np.squeeze(pred)
    cls_source = np.argmax(pred)
    score_source_org = pred.max()
    name_source = model.name_lookup.cls_to_name(cls_source,
                                                only_first_name=True)
    name_target = model.name_lookup.cls_to_name(cls_target,
                                                only_first_name=True)
    noise = 0
    for i in range(max_iterations):
        print("Iteration:", i)
        noisy_image = image + noise
        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)
        feed_dict = {model.tensor_name_resized_image: noisy_image,
                     pl_cls_target: cls_target}
        pred, grad = session.run([y_pred, gradient],
                                 feed_dict=feed_dict)
        pred = np.squeeze(pred)
        score_source = pred[cls_source]
        score_target = pred[cls_target]
        grad = np.array(grad).squeeze()
        grad_absmax = np.abs(grad).max()

        if grad_absmax < 1e-10:
            grad_absmax = 1e-10
        step_size = 7 / grad_absmax
        msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_source, cls_source, name_source))
        msg = "Target score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_target, cls_target, name_target))
        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
        print(msg.format(grad.min(), grad.max(), step_size))
        print()
        if score_target < required_score:
            noise -= step_size * grad
            noise = np.clip(a=noise,
                            a_min=-noise_limit,
                            a_max=noise_limit)
        else:
            break
    return image.squeeze(), noisy_image.squeeze(), noise,            name_source, name_target,            score_source, score_source_org, score_target

# ### Helper-function for plotting image and noise  # In[13]:
def normalize_image(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm  # In[14]:
def plot_images(image, noise, noisy_image,
                name_source, name_target,
                score_source, score_source_org, score_target):
    """
    Plot the image, the noisy image and the noise.
    Also shows the class-names and scores.
    Note that the noise is amplified to use the full range of
    colours, otherwise if the noise is very low it would be
    hard to see.
    image: Original input image.
    noise: Noise that has been added to the image.
    noisy_image: Input image + noise.
    name_source: Name of the source-class.
    name_target: Name of the target-class.
    score_source: Score for the source-class.
    score_source_org: Original score for the source-class.
    score_target: Score for the target-class.
    """

    fig, axes = plt.subplots(1, 3, figsize=(10,10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    smooth = True

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    ax = axes.flat[0]
    ax.imshow(image / 255.0, interpolation=interpolation)
    msg = "Original Image:\n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)
    ax = axes.flat[1]
    ax.imshow(noisy_image / 255.0, interpolation=interpolation)
    msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_source, name_target, score_target)
    ax.set_xlabel(xlabel)
    ax = axes.flat[2]
    ax.imshow(normalize_image(noise), interpolation=interpolation)
    xlabel = "Amplified Noise"
    ax.set_xlabel(xlabel)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    if is_plot: plt.show()

# ### Helper-function for finding and plotting adversarial example  # In[15]:
def adversary_example(image_path, cls_target,
                      noise_limit, required_score):
    """
    Find and plot adversarial noise for the given image.
    image_path: File-path to the input-image (must be *.jpg).
    cls_target: Target class-number (integer between 1-1000).
    noise_limit: Limit for pixel-values in the noise.
    required_score: Stop when target-class score reaches this.
    """
    image, noisy_image, noise,     name_source, name_target,     score_source, score_source_org, score_target =         find_adversary_noise(image_path=image_path,
                             cls_target=cls_target,
                             noise_limit=noise_limit,
                             required_score=required_score)
    plot_images(image=image, noise=noise, noisy_image=noisy_image,
                name_source=name_source, name_target=name_target,
                score_source=score_source,
                score_source_org=score_source_org,
                score_target=score_target)
    msg = "Noise min: {0:.3f}, max: {1:.3f}, mean: {2:.3f}, std: {3:.3f}"
    print(msg.format(noise.min(), noise.max(),
                     noise.mean(), noise.std()))

# ## Results

# ### Parrot  # In[16]:
image_path = "data/images/parrot_cropped1.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# ### Elon Musk  # In[17]:
image_path = "data/images/elon_musk.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# ### Willy Wonka (New)  # In[18]:
image_path = "data/images/willy_wonka_new.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# ### Willy Wonka (Old)  # In[19]:
image_path = "data/images/willy_wonka_old.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# ## Close TensorFlow Session  # In[20]:
print_time_usage(start_time_global)

# ## Conclusion

# ## Exercises
