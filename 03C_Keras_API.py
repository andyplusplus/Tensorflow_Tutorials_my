# # TensorFlow Tutorial #03-C

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import matplotlib.pyplot as plt
import tensorflow as tf
is_plot = False
import numpy as np
import math


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten


from mnist import MNIST

is_plot = False

data = MNIST(data_dir="data/MNIST/")

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))


img_size = data.img_size  #28
img_size_flat = data.img_size_flat  #784
img_shape = data.img_shape  #28, 28

img_shape_full = data.img_shape_full  #28, 28, 1
num_classes = data.num_classes     #10
num_channels = data.num_channels   #1

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
model.add(InputLayer(input_shape=(img_size_flat,)))  #784
model.add(Reshape(img_shape_full))    #(28, 28, 1)

model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1')) # First convolutional layer with ReLU-activation and max-pooling.
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2')) # Second convolutional layer with ReLU-activation and max-pooling.
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten()) # Flatten the 4-rank output of the convolutional layers # to 2-rank that can be input to a fully-connected / dense layer.
model.add(Dense(128, activation='relu')) # First fully-connected / dense layer with ReLU-activation.
model.add(Dense(num_classes, activation='softmax')) # Last fully-connected / dense layer with softmax-activation for use in classification.



# In[12]: # ### Model Compilation # The Neural Network has now been defined and must be finalized by adding a loss-function, optimizer and performance metrics. This is called model "compilation" in Keras. # We can either define the optimizer using a string, or if we want more control of its parameters then we need to instantiate an object. For example, we can set the learning-rate.
from tensorflow.python.keras.optimizers import Adam
optimizer = Adam(lr=1e-3)

# In[13]: # For a classification-problem such as MNIST which has 10 possible classes, we need to use the loss-function called `categorical_crossentropy`. The performance metric we are interested in is the classification accuracy.
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[14]: # ### Training # Now that the model has been fully defined with loss-function and optimizer, we can train it. This function takes numpy-arrays and performs the given number of training epochs using the given batch-size. An epoch is one full use of the entire training-set. So for 10 epochs we would iterate randomly over the entire training-set 10 times.
model.fit(x=data.x_train,
          y=data.y_train,
          epochs=1, batch_size=128)


# In[15]: # ### Evaluation # Now that the model has been trained we can test its performance on the test-set. This also uses numpy-arrays as input.
result = model.evaluate(x=data.x_test,
                        y=data.y_test)
for name, value in zip(model.metrics_names, result): # In[16]: # We can print all the performance metrics for the test-set.
    print(name, value)
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1])) # In[17]: # Or we can just print the classification accuracy.


# In[18]: # ### Prediction # We can also predict the classification for new images. We will just use some images from the test-set but you could load your own images into numpy arrays and use those instead.
images = data.x_test[0:9]
y_pred = model.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)

cls_true = data.y_test_cls[0:9]
plot_images(images=images,
            cls_true=cls_true,
            cls_pred=cls_pred)

# In[23]: # ### Examples of Mis-Classified Images
y_pred = model.predict(x=data.x_test)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)








# In[26]: # ## Functional Model
inputs = Input(shape=(img_size_flat,))
net = inputs

net = Reshape(img_shape_full)(net)
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

# Last fully-connected / dense layer with softmax-activation # so it can be used for classification.
net = Dense(num_classes, activation='softmax')(net)

# Output of the Neural Network.
outputs = net





# In[27]: # ### Model Compilation
from tensorflow.python.keras.models import Model
model2 = Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Training
model2.fit(x=data.x_train,
           y=data.y_train,
           epochs=1, batch_size=128)

# In[31]: # ### Evaluation
result = model2.evaluate(x=data.x_test,
                         y=data.y_test)

for name, value in zip(model2.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model2.metrics_names[1], result[1]))

# In[34]: # ### Examples of Mis-Classified Images
y_pred = model2.predict(x=data.x_test)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)


# In[37]: # ## Save & Load Model
path_model = 'model.keras'
model2.save(path_model)
del model2


# In[40]: # We need to import this Keras function for loading the model.
from tensorflow.python.keras.models import load_model
model3 = load_model(path_model)
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
y_pred = model3.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)

# In[46]: # Plot the images with their true and predicted class-numbers.
plot_images(images=images,
            cls_pred=cls_pred,
            cls_true=cls_true)



# In[47]: # ## Visualization of Layer Weights and Outputs
def plot_conv_weights(weights, input_channel=0):
    w_min = np.min(weights)
    w_max = np.max(weights)
    num_filters = weights.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = weights[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    
    if is_plot:
        if is_plot: plt.show()



# In[48]: # ### Get Layers # Keras has a simple way of listing the layers in the model.
model3.summary()
layer_input = model3.layers[0]

# In[50]: # The first convolutional layer has index 2.
layer_conv1 = model3.layers[2]
layer_conv1

# In[51]: # The second convolutional layer has index 4.
layer_conv2 = model3.layers[4]



# In[52]: # ### Convolutional Weights
weights_conv1 = layer_conv1.get_weights()[0]
weights_conv1.shape
plot_conv_weights(weights=weights_conv1, input_channel=0)

weights_conv2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights=weights_conv2, input_channel=0)


# In[57]: # ### Helper-function for plotting the output of a convolutional layer
def plot_conv_output(values):
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    if is_plot:
        if is_plot: plt.show()



# In[58]: # ### Input Image # Helper-function for plotting a single image.
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    if is_plot:
        if is_plot: plt.show()

# In[59]: # Plot an image from the test-set which will be used as an example below.
image1 = data.x_test[0]
plot_image(image1)




# In[60]: # ### Output of Convolutional Layer - Method 1
# There are different ways of getting the output of a layer in a Keras model. This method uses a so-called K-function which turns a part of the Keras model into a function.

from tensorflow.python.keras import backend as K

output_conv1 = K.function(inputs=[layer_input.input],
                          outputs=[layer_conv1.output])

layer_output1 = output_conv1([[image1]])[0]   #(1, 28, 28, 16)
layer_output1.shape  #(1, 28, 28, 16)

plot_conv_output(values=layer_output1)





# In[64]: # ### Output of Convolutional Layer - Method 2 # Keras also has another method for getting the output of a layer inside the model. This creates another Functional Model using the same input as the original model, but the output is now taken from the convolutional layer that we are interested in.
output_conv2 = Model(inputs=layer_input.input,
                     outputs=layer_conv2.output)

layer_output2 = output_conv2.predict(np.array([image1]))
layer_output2.shape   #(1, 14, 14, 36)

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
