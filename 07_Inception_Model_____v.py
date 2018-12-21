# # TensorFlow Tutorial #07

# # Inception Model

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
import os
import inception  # In[3]:
tf.__version__

# ## Download the Inception Model  # In[4]:  # In[5]:
inception.maybe_download()

# ## Load the Inception Model  # In[6]:
model = inception.Inception()

# ## Helper-function for classifying and plotting images  # In[7]:
def classify(image_path):
    pred = model.classify(image_path=image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)    

# ## Panda  # In[8]:
image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path)

# ## Interpretation of Classification Scores

# ## Parrot (Original Image)  # In[9]:
classify(image_path="data/images/parrot.jpg")

# ## Parrot (Resized Image)  # In[10]:
def plot_resized_image(image_path):
    resized_image = model.get_resized_image(image_path=image_path)
    plt.imshow(resized_image, interpolation='nearest')

    if is_plot: plt.show()  # In[11]:
plot_resized_image(image_path="data/images/parrot.jpg")

# ## Parrot (Cropped Image, Top)  # In[12]:
classify(image_path="data/images/parrot_cropped1.jpg")

# ## Parrot (Cropped Image, Middle)  # In[13]:
classify(image_path="data/images/parrot_cropped2.jpg")

# ## Parrot (Cropped Image, Bottom)  # In[14]:
classify(image_path="data/images/parrot_cropped3.jpg")

# ## Parrot (Padded Image)  # In[15]:
classify(image_path="data/images/parrot_padded.jpg")

# ## Elon Musk (299 x 299 pixels)  # In[16]:
classify(image_path="data/images/elon_musk.jpg")

# ## Elon Musk (100 x 100 pixels)  # In[17]:
classify(image_path="data/images/elon_musk_100x100.jpg")  # In[18]:
plot_resized_image(image_path="data/images/elon_musk_100x100.jpg")

# ## Willy Wonka (Gene Wilder)  # In[19]:
classify(image_path="data/images/willy_wonka_old.jpg")

# ## Willy Wonka (Johnny Depp)  # In[20]:
classify(image_path="data/images/willy_wonka_new.jpg")

# ## Close TensorFlow Session  # In[21]:

# ## Conclusion

# ## Exercises

# ## License (MIT)
