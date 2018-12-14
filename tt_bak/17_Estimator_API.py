#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Tutorial #17
# # Estimator API
# 
# by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
# / [GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)

# ## Introduction
# 
# High-level API's are extremely important in all software development because they provide simple abstractions for doing very complicated tasks. This makes it easier to write and understand your source-code, and it lowers the risk of errors.
# 
# In Tutorial #03 we saw how to use various builder API's for creating Neural Networks in TensorFlow. However, there was a lot of additional code required for training the models and using them on new data. The Estimator is another high-level API that implements most of this, although it can be debated how simple it really is.
# 
# Using the Estimator API consists of several steps:
# 
# 1. Define functions for inputting data to the Estimator.
# 2. Either use an existing Estimator (e.g. a Deep Neural Network), which is also called a pre-made or Canned Estimator. Or create your own Estimator, in which case you also need to define the optimizer, performance metrics, etc.
# 3. Train the Estimator using the training-set defined in step 1.
# 4. Evaluate the performance of the Estimator on the test-set defined in step 1.
# 5. Use the trained Estimator to make predictions on other data.

# ## Imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# This was developed using Python 3.6 (Anaconda) and TensorFlow version:

# In[2]:


tf.__version__


# ## Load Data

# The MNIST data-set is about 12 MB and will be downloaded automatically if it is not located in the given dir.

# In[3]:


from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")


# The MNIST data-set has now been loaded and consists of 70.000 images and class-numbers for the images. The data-set is split into 3 mutually exclusive sub-sets. We will only use the training and test-sets in this tutorial.

# In[4]:


print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))


# Copy some of the data-dimensions for convenience.

# In[5]:


# The number of pixels in each dimension of an image.
img_size = data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels


# ### Helper-function for plotting images

# Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.

# In[6]:


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# ### Plot a few images to see if data is correct

# In[7]:


# Get the first images from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# ## Input Functions for the Estimator

# Rather than providing raw data directly to the Estimator, we must provide functions that return the data. This allows for more flexibility in data-sources and how the data is randomly shuffled and iterated.
# 
# Note that we will create an Estimator using the `DNNClassifier` which assumes the class-numbers are integers so we use `data.y_train_cls` instead of `data.y_train` which are one-hot encoded arrays.
# 
# The function also has parameters for `batch_size`, `queue_capacity` and `num_threads` for finer control of the data reading. In our case we take the data directly from a numpy array in memory, so it is not needed.

# In[8]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.x_train)},
    y=np.array(data.y_train_cls),
    num_epochs=None,
    shuffle=True)


# This actually returns a function:

# In[9]:


train_input_fn


# Calling this function returns a tuple with TensorFlow ops for returning the input and output data:

# In[10]:


train_input_fn()


# Similarly we need to create a function for reading the data for the test-set. Note that we only want to process these images once so `num_epochs=1` and we do not want the images shuffled so `shuffle=False`.

# In[11]:


test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.x_test)},
    y=np.array(data.y_test_cls),
    num_epochs=1,
    shuffle=False)


# An input-function is also needed for predicting the class of new data. As an example we just use a few images from the test-set.

# In[12]:


some_images = data.x_test[0:9]


# In[13]:


predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": some_images},
    num_epochs=1,
    shuffle=False)


# The class-numbers are actually not used in the input-function as it is not needed for prediction. However, the true class-number is needed when we plot the images further below.

# In[14]:


some_images_cls = data.y_test_cls[0:9]


# ## Pre-Made / Canned Estimator
# 
# When using a pre-made Estimator, we need to specify the input features for the data. In this case we want to input images from our data-set which are numeric arrays of the given shape.

# In[15]:


feature_x = tf.feature_column.numeric_column("x", shape=img_shape)


# You can have several input features which would then be combined in a list:

# In[16]:


feature_columns = [feature_x]


# In this example we want to use a 3-layer DNN with 512, 256 and 128 units respectively.

# In[17]:


num_hidden_units = [512, 256, 128]


# The `DNNClassifier` then constructs the neural network for us. We can also specify the activation function and various other parameters (see the docs). Here we just specify the number of classes and the directory where the checkpoints will be saved.

# In[18]:


model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=num_hidden_units,
                                   activation_fn=tf.nn.relu,
                                   n_classes=num_classes,
                                   model_dir="./checkpoints_tutorial17-1/")


# ### Training
# 
# We can now train the model for a given number of iterations. This automatically loads and saves checkpoints so we can continue the training later.
# 
# Note that the text `INFO:tensorflow:` is printed on every line and makes it harder to quickly read the actual progress. It should have been printed on a single line instead.

# In[19]:


model.train(input_fn=train_input_fn, steps=2000)


# ### Evaluation
# 
# Once the model has been trained, we can evaluate its performance on the test-set.

# In[20]:


result = model.evaluate(input_fn=test_input_fn)


# In[21]:


result


# In[22]:


print("Classification accuracy: {0:.2%}".format(result["accuracy"]))


# ### Predictions
# 
# The trained model can also be used to make predictions on new data.
# 
# Note that the TensorFlow graph is recreated and the checkpoint is reloaded every time we make predictions on new data. If the model is very large then this could add a significant overhead.
# 
# It is unclear why the Estimator is designed this way, possibly because it will always use the latest checkpoint and it can also be distributed easily for use on multiple computers.

# In[23]:


predictions = model.predict(input_fn=predict_input_fn)


# In[24]:


cls = [p['classes'] for p in predictions]


# In[25]:


cls_pred = np.array(cls, dtype='int').squeeze()
cls_pred


# In[26]:


plot_images(images=some_images,
            cls_true=some_images_cls,
            cls_pred=cls_pred)


# # New Estimator

# If you cannot use one of the built-in Estimators, then you can create an arbitrary TensorFlow model yourself. To do this, you first need to create a function which defines the following:
# 
# 1. The TensorFlow model, e.g. a Convolutional Neural Network.
# 2. The output of the model.
# 3. The loss-function used to improve the model during optimization.
# 4. The optimization method.
# 5. Performance metrics.
# 
# The Estimator can be run in three modes: Training, Evaluation, or Prediction. The code is mostly the same, but in Prediction-mode we do not need to setup the loss-function and optimizer.
# 
# This is another aspect of the Estimator API that is poorly designed and resembles how we did ANSI C programming using structs in the old days. It would probably have been more elegant to split this into several functions and sub-classed the Estimator-class.

# In[27]:


def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn,
    #           see e.g. train_input_fn for these two.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.
    
    # Reference to the tensor named "x" in the input-function.
    x = features["x"]

    # The convolutional layers expect 4-rank tensors
    # but x is a 2-rank tensor, so reshape it.
    net = tf.reshape(x, [-1, img_size, img_size, num_channels])    

    # First convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=16, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=36, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)    

    # Flatten to a 2-rank tensor.
    net = tf.contrib.layers.flatten(net)
    # Eventually this should be replaced with:
    # net = tf.layers.flatten(net)

    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)    

    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc2',
                          units=10)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)
    
    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.
        
        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics =         {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec


# ### Create an Instance of the Estimator
# 
# We can specify hyper-parameters e.g. for the learning-rate of the optimizer.

# In[28]:


params = {"learning_rate": 1e-4}


# We can then create an instance of the new Estimator.
# 
# Note that we don't provide feature-columns here as it is inferred automatically from the data-functions when `model_fn()` is called.
# 
# It is unclear from the TensorFlow documentation why it is necessary to specify the feature-columns when using `DNNClassifier` in the example above, when it is not needed here.

# In[29]:


model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="./checkpoints_tutorial17-2/")


# ### Training
# 
# Now that our new Estimator has been created, we can train it.

# In[30]:


model.train(input_fn=train_input_fn, steps=2000)


# ### Evaluation
# 
# Once the model has been trained, we can evaluate its performance on the test-set.

# In[31]:


result = model.evaluate(input_fn=test_input_fn)


# In[32]:


result


# In[33]:


print("Classification accuracy: {0:.2%}".format(result["accuracy"]))


# ### Predictions
# 
# The model can also be used to make predictions on new data.

# In[34]:


predictions = model.predict(input_fn=predict_input_fn)


# In[35]:


cls_pred = np.array(list(predictions))
cls_pred


# In[36]:


plot_images(images=some_images,
            cls_true=some_images_cls,
            cls_pred=cls_pred)


# ## Conclusion
# 
# This tutorial showed how to use the Estimator API in TensorFlow. It is supposed to make it easier to train and use a model, but it seems to have several design problems:
# 
# * The Estimator API is complicated, inconsistent and confusing.
# * Error-messages are extremely long and often impossible to understand.
# * The TensorFlow graph is recreated and the checkpoint is reloaded EVERY time you want to use a trained model to make a prediction on new data. Some models are very big so this could add a very large overhead. A better way might be to only reload the model if the checkpoint has changed on disk.
# * It is unclear how to gain access to the trained model, e.g. to plot the weights of a neural network.
# 
# It seems that the Estimator API could have been much simpler and easier to use. For small projects you may find it too complicated and confusing to be worth the effort. But it is possible that the Estimator API is useful if you have a very large dataset and if you train on many machines.

# ## Exercises
# 
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# 
# You may want to backup this Notebook before making any changes.
# 
# * Run another 10000 training iterations for each model.
# * Print classification accuracy on the test-set before optimization and after 1000, 2000 and 10000 iterations.
# * Change the structure of the neural network inside the Estimator. Do you have to delete the checkpoint-files? Why?
# * Change the batch-size for the input-functions.
# * In many of the previous tutorials we plotted examples of mis-classified images. Do that here as well.
# * Change the Estimator to use one-hot encoded labels instead of integer class-numbers.
# * Change the input-functions to load image-files instead of using numpy-arrays.
# * Can you find a way to plot the weights of the neural network and the output of the individual layers?
# * List 5 things you like and don't like about the Estimator API. Do you have any suggestions for improvements? Maybe you should suggest them to the developers?
# * Explain to a friend how the program works.

# ## License (MIT)
# 
# Copyright (c) 2016-2017 by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
