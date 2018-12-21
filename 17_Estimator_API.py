# # TensorFlow Tutorial #17

# # Estimator API

# ## Introduction

# ## Imports  # In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np  # In[2]:
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
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
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

# ## Input Functions for the Estimator  # In[8]:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.x_train)},
    y=np.array(data.y_train_cls),
    num_epochs=None,
    shuffle=True)  # In[9]:
train_input_fn  # In[10]:
train_input_fn()  # In[11]:
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.x_test)},
    y=np.array(data.y_test_cls),
    num_epochs=1,
    shuffle=False)  # In[12]:
some_images = data.x_test[0:9]  # In[13]:
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": some_images},
    num_epochs=1,
    shuffle=False)  # In[14]:
some_images_cls = data.y_test_cls[0:9]

# ## Pre-Made / Canned Estimator  # In[15]:
feature_x = tf.feature_column.numeric_column("x", shape=img_shape)  # In[16]:
feature_columns = [feature_x]  # In[17]:
num_hidden_units = [512, 256, 128]  # In[18]:
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=num_hidden_units,
                                   activation_fn=tf.nn.relu,
                                   n_classes=num_classes,
                                   model_dir="./checkpoints_tutorial17-1/")

# ### Training  # In[19]:
model.train(input_fn=train_input_fn, steps=2000)

# ### Evaluation  # In[20]:
result = model.evaluate(input_fn=test_input_fn)  # In[21]:
result  # In[22]:
print("Classification accuracy: {0:.2%}".format(result["accuracy"]))

# ### Predictions  # In[23]:
predictions = model.predict(input_fn=predict_input_fn)  # In[24]:
cls = [p['classes'] for p in predictions]  # In[25]:
cls_pred = np.array(cls, dtype='int').squeeze()
cls_pred  # In[26]:
plot_images(images=some_images,
            cls_true=some_images_cls,
            cls_pred=cls_pred)

# # New Estimator  # In[27]:
def model_fn(features, labels, mode, params):
    
    x = features["x"]
    net = tf.reshape(x, [-1, img_size, img_size, num_channels])    
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=16, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=36, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)    
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)    
    net = tf.layers.dense(inputs=net, name='layer_fc2',
                          units=10)
    logits = net
    y_pred = tf.nn.softmax(logits=logits)
    
    y_pred_cls = tf.argmax(y_pred, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        metrics =         {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec

# ### Create an Instance of the Estimator  # In[28]:
params = {"learning_rate": 1e-4}  # In[29]:
model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="./checkpoints_tutorial17-2/")

# ### Training  # In[30]:
model.train(input_fn=train_input_fn, steps=2000)

# ### Evaluation  # In[31]:
result = model.evaluate(input_fn=test_input_fn)  # In[32]:
result  # In[33]:
print("Classification accuracy: {0:.2%}".format(result["accuracy"]))

# ### Predictions  # In[34]:
predictions = model.predict(input_fn=predict_input_fn)  # In[35]:
cls_pred = np.array(list(predictions))
cls_pred  # In[36]:
plot_images(images=some_images,
            cls_true=some_images_cls,
            cls_pred=cls_pred)

# ## Conclusion

# ## Exercises

# ## License (MIT)
