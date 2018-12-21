# # TensorFlow Tutorial #19

# # Hyper-Parameter Optimization

# ## Introduction

# ## Flowchart

# ## Imports  # In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np
import math  # In[2]:
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model  # In[3]:
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args  # In[4]:
tf.__version__  # In[5]:
tf.keras.__version__  # In[6]:
skopt.__version__

# ## Hyper-Parameters  # In[7]:
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')  # In[8]:
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')  # In[9]:
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')  # In[10]:
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')  # In[11]:
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation]  # In[12]:
default_parameters = [1e-5, 1, 16, 'relu']

# ### Helper-function for log-dir-name  # In[13]:
def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation)
    return log_dir

# ## Load Data  # In[14]:
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)  # In[15]:
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))  # In[16]:
data.test.cls = np.argmax(data.test.labels, axis=1)  # In[17]:
validation_data = (data.validation.images, data.validation.labels)

# ## Data Dimensions  # In[18]:
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
img_shape_full = (img_size, img_size, 1)
num_channels = 1
num_classes = 10

# ### Helper-function for plotting images  # In[19]:
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

# ### Plot a few images to see if data is correct  # In[20]:
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)

# ### Helper-function to plot example errors  # In[21]:
def plot_example_errors(cls_pred):
    incorrect = (cls_pred != data.test.cls)
    images = data.test.images[incorrect]
    
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# ## Hyper-Parameter Optimization

# ### Create the Model  # In[22]:
def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size_flat,)))
    model.add(Reshape(img_shape_full))
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation=activation, name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation=activation, name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(lr=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# ### Train and Evaluate the Model  # In[23]:
path_best_model = '19_best_model.keras'  # In[24]:
best_accuracy = 0.0  # In[25]:
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print()
    
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation)
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, activation)
    
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)
   
    history = model.fit(x=data.train.images,
                        y=data.train.labels,
                        epochs=3,
                        batch_size=128,
                        validation_data=validation_data,
                        callbacks=[callback_log])
    accuracy = history.history['val_acc'][-1]
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()
    global best_accuracy
    if accuracy > best_accuracy:
        model.save(path_best_model)
        
        best_accuracy = accuracy
    del model
    
    K.clear_session()
    
    return -accuracy

# ### Test Run  # In[26]:
fitness(x=default_parameters)

# ### Run the Hyper-Parameter Optimization  # In[27]:
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)

# ### Optimization Progress  # In[28]:
plot_convergence(search_result)

# ### Best Hyper-Parameters  # In[29]:
search_result.x  # In[30]:
space = search_result.space  # In[31]:
space.point_to_dict(search_result.x)  # In[32]:
search_result.fun  # In[33]:
sorted(zip(search_result.func_vals, search_result.x_iters))

# ### Plots  # In[34]:
fig, ax = plot_histogram(result=search_result,
                         dimension_name='activation')  # In[35]:
fig = plot_objective_2D(result=search_result,
                        dimension_name1='learning_rate',
                        dimension_name2='num_dense_layers',
                        levels=50)  # In[36]:
dim_names = ['learning_rate', 'num_dense_nodes', 'num_dense_layers']  # In[37]:
fig, ax = plot_objective(result=search_result, dimension_names=dim_names)  # In[38]:
fig, ax = plot_evaluations(result=search_result, dimension_names=dim_names)

# ### Evaluate Best Model on Test-Set  # In[39]:
model = load_model(path_best_model)  # In[40]:
result = model.evaluate(x=data.test.images,
                        y=data.test.labels)  # In[41]:
for name, value in zip(model.metrics_names, result):
    print(name, value)  # In[42]:
print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

# ### Predict on New Data  # In[43]:
images = data.test.images[0:9]  # In[44]:
cls_true = data.test.cls[0:9]  # In[45]:
y_pred = model.predict(x=images)  # In[46]:
cls_pred = np.argmax(y_pred,axis=1)  # In[47]:
plot_images(images=images,
            cls_true=cls_true,
            cls_pred=cls_pred)

# ### Examples of Mis-Classified Images  # In[48]:
y_pred = model.predict(x=data.test.images)  # In[49]:
cls_pred = np.argmax(y_pred,axis=1)  # In[50]:
plot_example_errors(cls_pred)

# ## Conclusion

# ## Exercises

# ## License (MIT)
