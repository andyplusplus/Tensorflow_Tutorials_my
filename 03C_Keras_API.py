# # TensorFlow Tutorial #03-C

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten


from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))


img_size = data.img_size
img_size_flat = data.img_size_flat
img_shape = data.img_shape

img_shape_full = data.img_shape_full
num_classes = data.num_classes
num_channels = data.num_channels

# In[7]: # ### Helper-function for plotting images
# from common.plot_helper import plot_images

def plot_images(images, cls_true, cls_pred=None, img_shape=(28,28)):
    pass


images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
plot_images(images=images, cls_true=cls_true)

# In[9]: # ### Helper-function to plot example errors # Function for plotting examples of images from the test-set that have been mis-classified.
def plot_example_errors(cls_pred):
    incorrect = (cls_pred != data.y_test_cls)
    images = data.x_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.y_test_cls[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# In[10]: # ## PrettyTensor API # This is how the Convolutional Neural Network was implemented in Tutorial #03 using the PrettyTensor API. It is shown here for easy comparison to the Keras implementation below.
if False:
    x_pretty = pt.wrap(x_image)

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.            conv2d(kernel=5, depth=16, name='layer_conv1').            max_pool(kernel=2, stride=2).            conv2d(kernel=5, depth=36, name='layer_conv2').            max_pool(kernel=2, stride=2).            flatten().            fully_connected(size=128, name='layer_fc1').            softmax_classifier(num_classes=num_classes, labels=y_true)

# In[11]: # ## Sequential Model # The Keras API has two modes of constructing Neural Networks. The simplest is the Sequential Model which only allows for the layers to be added in sequence.
model = Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size.
model.add(InputLayer(input_shape=(img_size_flat,)))

# The input is a flattened array with 784 elements,
# but the convolutional layers expect images with shape (28, 28, 1)
model.add(Reshape(img_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the 4-rank output of the convolutional layers
# to 2-rank that can be input to a fully-connected / dense layer.
model.add(Flatten())

# First fully-connected / dense layer with ReLU-activation.
model.add(Dense(128, activation='relu'))

# Last fully-connected / dense layer with softmax-activation
# for use in classification.
model.add(Dense(num_classes, activation='softmax'))

# ### Model Compilation
# The Neural Network has now been defined and must be finalized by adding a loss-function, optimizer and performance metrics. This is called model "compilation" in Keras.
# We can either define the optimizer using a string, or if we want more control of its parameters then we need to instantiate an object. For example, we can set the learning-rate.

# In[12]:
from tensorflow.python.keras.optimizers import Adam

optimizer = Adam(lr=1e-3)

# For a classification-problem such as MNIST which has 10 possible classes, we need to use the loss-function called `categorical_crossentropy`. The performance metric we are interested in is the classification accuracy.

# In[13]:
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ### Training
# Now that the model has been fully defined with loss-function and optimizer, we can train it. This function takes numpy-arrays and performs the given number of training epochs using the given batch-size. An epoch is one full use of the entire training-set. So for 10 epochs we would iterate randomly over the entire training-set 10 times.

# In[14]:
model.fit(x=data.x_train,
          y=data.y_train,
          epochs=1, batch_size=128)

# ### Evaluation
# Now that the model has been trained we can test its performance on the test-set. This also uses numpy-arrays as input.

# In[15]:
result = model.evaluate(x=data.x_test,
                        y=data.y_test)

# We can print all the performance metrics for the test-set.

# In[16]:
for name, value in zip(model.metrics_names, result):
    print(name, value)

# Or we can just print the classification accuracy.

# In[17]:
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

# ### Prediction
# We can also predict the classification for new images. We will just use some images from the test-set but you could load your own images into numpy arrays and use those instead.

# In[18]:
images = data.x_test[0:9]

# These are the true class-number for those images. This is only used when plotting the images.

# In[19]:
cls_true = data.y_test_cls[0:9]

# Get the predicted classes as One-Hot encoded arrays.

# In[20]:
y_pred = model.predict(x=images)

# Get the predicted classes as integers.

# In[21]:
cls_pred = np.argmax(y_pred, axis=1)

# In[22]:
plot_images(images=images,
            cls_true=cls_true,
            cls_pred=cls_pred)

# ### Examples of Mis-Classified Images
# We can plot some examples of mis-classified images from the test-set.
# First we get the predicted classes for all the images in the test-set:

# In[23]:
y_pred = model.predict(x=data.x_test)

# Then we convert the predicted class-numbers from One-Hot encoded arrays to integers.

# In[24]:
cls_pred = np.argmax(y_pred, axis=1)

# Plot some of the mis-classified images.

# In[25]:
plot_example_errors(cls_pred)

# ## Functional Model
# The Keras API can also be used to construct more complicated networks using the Functional Model. This may look a little confusing at first, because each call to the Keras API will create and return an instance that is itself callable. It is not clear whether it is a function or an object - but we can call it as if it is a function. This allows us to build computational graphs that are more complex than the Sequential Model allows.

# In[26]:
# Create an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size.
inputs = Input(shape=(img_size_flat,))

# Variable used for building the Neural Network.
net = inputs

# The input is an image as a flattened array with 784 elements.
# But the convolutional layers expect images with shape (28, 28, 1)
net = Reshape(img_shape_full)(net)

# First convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
             activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Second convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
             activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Flatten the output of the conv-layer from 4-dim to 2-dim.
net = Flatten()(net)

# First fully-connected / dense layer with ReLU-activation.
net = Dense(128, activation='relu')(net)

# Last fully-connected / dense layer with softmax-activation
# so it can be used for classification.
net = Dense(num_classes, activation='softmax')(net)

# Output of the Neural Network.
outputs = net

# ### Model Compilation
# We have now defined the architecture of the model with its input and output. We now have to create a Keras model and compile it with a loss-function and optimizer, so it is ready for training.

# In[27]:
from tensorflow.python.keras.models import Model

# Create a new instance of the Keras Functional Model. We give it the inputs and outputs of the Convolutional Neural Network that we constructed above.

# In[28]:
model2 = Model(inputs=inputs, outputs=outputs)

# Compile the Keras model using the RMSprop optimizer and with a loss-function for multiple categories. The only performance metric we are interested in is the classification accuracy, but you could use a list of metrics here.

# In[29]:
model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# ### Training
# The model has now been defined and compiled so it can be trained using the same `fit()` function as used in the Sequential Model above. This also takes numpy-arrays as input.

# In[30]:
model2.fit(x=data.x_train,
           y=data.y_train,
           epochs=1, batch_size=128)

# ### Evaluation
# Once the model has been trained we can evaluate its performance on the test-set. This is the same syntax as for the Sequential Model.

# In[31]:
result = model2.evaluate(x=data.x_test,
                         y=data.y_test)

# The result is a list of values, containing the loss-value and all the metrics we defined when we compiled the model. Note that 'accuracy' is now called 'acc' which is a small inconsistency.

# In[32]:
for name, value in zip(model2.metrics_names, result):
    print(name, value)

# We can also print the classification accuracy as a percentage:

# In[33]:
print("{0}: {1:.2%}".format(model2.metrics_names[1], result[1]))

# ### Examples of Mis-Classified Images
# We can plot some examples of mis-classified images from the test-set.
# First we get the predicted classes for all the images in the test-set:

# In[34]:
y_pred = model2.predict(x=data.x_test)

# Then we convert the predicted class-numbers from One-Hot encoded arrays to integers.

# In[35]:
cls_pred = np.argmax(y_pred, axis=1)

# Plot some of the mis-classified images.

# In[36]:
plot_example_errors(cls_pred)

# ## Save & Load Model
# NOTE: You need to install `h5py` for this to work!
# Tutorial #04 was about saving and restoring the weights of a model using native TensorFlow code. It was an absolutely horrible API! Fortunately, Keras makes this very easy.
# This is the file-path where we want to save the Keras model.

# In[37]:
path_model = 'model.keras'

# Saving a Keras model with the trained weights is then just a single function call, as it should be.

# In[38]:
model2.save(path_model)

# Delete the model from memory so we are sure it is no longer used.

# In[39]:
del model2

# We need to import this Keras function for loading the model.

# In[40]:
from tensorflow.python.keras.models import load_model

# Loading the model is then just a single function-call, as it should be.

# In[41]:
model3 = load_model(path_model)

# We can then use the model again e.g. to make predictions. We get the first 9 images from the test-set and their true class-numbers.

# In[42]:
images = data.x_test[0:9]

# In[43]:
cls_true = data.y_test_cls[0:9]

# We then use the restored model to predict the class-numbers for those images.

# In[44]:
y_pred = model3.predict(x=images)

# Get the class-numbers as integers.

# In[45]:
cls_pred = np.argmax(y_pred, axis=1)

# Plot the images with their true and predicted class-numbers.

# In[46]:
plot_images(images=images,
            cls_pred=cls_pred,
            cls_true=cls_true)

# ## Visualization of Layer Weights and Outputs

# ### Helper-function for plotting convolutional weights

# In[47]:
def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# ### Get Layers
# Keras has a simple way of listing the layers in the model.

# In[48]:
model3.summary()

# We count the indices to get the layers we want.
# The input-layer has index 0.

# In[49]:
layer_input = model3.layers[0]

# The first convolutional layer has index 2.

# In[50]:
layer_conv1 = model3.layers[2]
layer_conv1

# The second convolutional layer has index 4.

# In[51]:
layer_conv2 = model3.layers[4]

# ### Convolutional Weights
# Now that we have the layers we can easily get their weights.

# In[52]:
weights_conv1 = layer_conv1.get_weights()[0]

# This gives us a 4-rank tensor.

# In[53]:
weights_conv1.shape

# Plot the weights using the helper-function from above.

# In[54]:
plot_conv_weights(weights=weights_conv1, input_channel=0)

# We can also get the weights for the second convolutional layer and plot them.

# In[55]:
weights_conv2 = layer_conv2.get_weights()[0]

# In[56]:
plot_conv_weights(weights=weights_conv2, input_channel=0)

# ### Helper-function for plotting the output of a convolutional layer

# In[57]:
def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# ### Input Image
# Helper-function for plotting a single image.

# In[58]:
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

# Plot an image from the test-set which will be used as an example below.

# In[59]:
image1 = data.x_test[0]
plot_image(image1)

# ### Output of Convolutional Layer - Method 1
# There are different ways of getting the output of a layer in a Keras model. This method uses a so-called K-function which turns a part of the Keras model into a function.

# In[60]:
from tensorflow.python.keras import backend as K

# In[61]:
output_conv1 = K.function(inputs=[layer_input.input],
                          outputs=[layer_conv1.output])

# We can then call this function with the input image. Note that the image is wrapped in two lists because the function expects an array of that dimensionality. Likewise, the function returns an array with one more dimensionality than we want so we just take the first element.

# In[62]:
layer_output1 = output_conv1([[image1]])[0]
layer_output1.shape

# We can then plot the output of all 16 channels of the convolutional layer.

# In[63]:
plot_conv_output(values=layer_output1)

# ### Output of Convolutional Layer - Method 2
# Keras also has another method for getting the output of a layer inside the model. This creates another Functional Model using the same input as the original model, but the output is now taken from the convolutional layer that we are interested in.

# In[64]:
output_conv2 = Model(inputs=layer_input.input,
                     outputs=layer_conv2.output)

# This creates a new model-object where we can call the typical Keras functions. To get the output of the convoloutional layer we call the `predict()` function with the input image.

# In[65]:
layer_output2 = output_conv2.predict(np.array([image1]))
layer_output2.shape

# We can then plot the images for all 36 channels.

# In[66]:
plot_conv_output(values=layer_output2)

# ## Conclusion

# ## Exercises
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# You may want to backup this Notebook before making any changes.
# * Train for more epochs. Does it improve the classification accuracy?
# * Change the activation function to sigmoid for some of the layers.
# * Can you find a simple way of changing the activation function for all the layers?
# * Plot the output of the max-pooling layers instead of the conv-layers.
# * Replace the 2x2 max-pooling layers with stride=2 in the convolutional layers. Is there a difference in classification accuracy? What if you optimize it again and again? The difference is random, so how would you measure if there really is a difference? What are the pros and cons of using max-pooling vs. stride in the conv-layer?
# * Change the parameters for the layers, e.g. the kernel, depth, size, etc. What is the difference in time usage and classification accuracy?
# * Add and remove some convolutional and fully-connected layers.
# * What is the simplest network you can design that still performs well?
# * Change the Functional Model so it has another convolutional layer that connects in parallel to the existing conv-layers before going into the dense layers.
# * Change the Functional Model so it outputs the predicted class both as a One-Hot encoded array and as an integer, so we don't have to use `numpy.argmax()` afterwards.
# * Remake the program yourself without looking too much at this source-code.
# * Explain to a friend how the program works.

