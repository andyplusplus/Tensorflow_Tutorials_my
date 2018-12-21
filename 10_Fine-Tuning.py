# # TensorFlow Tutorial #10

# # Fine-Tuning

# ## Introduction

# ## Flowchart

# ## Imports  # In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np
import os  # In[2]:
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop  # In[3]:
tf.__version__

# ## Helper Functions

# ### Helper-function for joining a directory and list of filenames.  # In[4]:
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

# ### Helper-function for plotting images  # In[5]:
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(3, 3)
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i],
                      interpolation=interpolation)
            cls_true_name = class_names[cls_true[i]]
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                cls_pred_name = class_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    if is_plot: plt.show()

# ### Helper-function for printing confusion matrix  # In[6]:
from sklearn.metrics import confusion_matrix
def print_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.
    print("Confusion matrix:")
    
    print(cm)
    
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))

# ### Helper-function for plotting example errors  # In[7]:
def plot_example_errors(cls_pred):
    incorrect = (cls_pred != cls_test)
    image_paths = np.array(image_paths_test)[incorrect]
    images = load_images(image_paths=image_paths[0:9])
    
    cls_pred = cls_pred[incorrect]
    cls_true = cls_test[incorrect]
    
    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])  # In[8]:
def example_errors():
    generator_test.reset()
    
    y_pred = new_model.predict_generator(generator_test,
                                         steps=steps_test)
    cls_pred = np.argmax(y_pred,axis=1)
    plot_example_errors(cls_pred)
    
    print_confusion_matrix(cls_pred)

# ### Helper-function for loading images  # In[9]:
def load_images(image_paths):
    images = [plt.imread(path) for path in image_paths]
    return np.asarray(images)

# ### Helper-function for plotting training history  # In[10]:
def plot_training_history(history):
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')
    
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')
    plt.title('Training and Test Accuracy')
    plt.legend()
    if is_plot: plt.show()

# ## Dataset: Knifey-Spoony  # In[11]:
import knifey  # In[12]:
knifey.maybe_download_and_extract()  # In[13]:
knifey.copy_files()  # In[14]:
train_dir = knifey.train_dir
test_dir = knifey.test_dir

# ## Pre-Trained Model: VGG16  # In[15]:
model = VGG16(include_top=True, weights='imagenet')

# ## Input Pipeline  # In[16]:
input_shape = model.layers[0].output_shape[1:3]
input_shape  # In[17]:
datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')  # In[18]:
datagen_test = ImageDataGenerator(rescale=1./255)  # In[19]:
batch_size = 20  # In[20]:
if True:
    save_to_dir = None
else:
    save_to_dir='augmented_images/'  # In[21]:
generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)  # In[22]:
generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)  # In[23]:
steps_test = generator_test.n / batch_size
steps_test  # In[24]:
image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)  # In[25]:
cls_train = generator_train.classes
cls_test = generator_test.classes  # In[26]:
class_names = list(generator_train.class_indices.keys())
class_names  # In[27]:
num_classes = generator_train.num_classes
num_classes

# ### Plot a few images to see if data is correct  # In[28]:
images = load_images(image_paths=image_paths_train[0:9])
cls_true = cls_train[0:9]
plot_images(images=images, cls_true=cls_true, smooth=True)

# ### Class Weights  # In[29]:
from sklearn.utils.class_weight import compute_class_weight  # In[30]:
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)  # In[31]:
class_weight  # In[32]:
class_names

# ## Example Predictions  # In[33]:
def predict(image_path):
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)
    plt.imshow(img_resized)
    if is_plot: plt.show()
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    pred = model.predict(img_array)
    
    pred_decoded = decode_predictions(pred)[0]
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))  # In[34]:
predict(image_path='images/parrot_cropped1.jpg')  # In[35]:
predict(image_path=image_paths_train[0])  # In[36]:
predict(image_path=image_paths_train[1])  # In[37]:
predict(image_path=image_paths_test[0])

# ## Transfer Learning  # In[38]:
model.summary()  # In[39]:
transfer_layer = model.get_layer('block5_pool')  # In[40]:
transfer_layer.output  # In[41]:
conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)  # In[42]:
new_model = Sequential()
new_model.add(conv_model)
new_model.add(Flatten())
new_model.add(Dense(1024, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(num_classes, activation='softmax'))  # In[43]:
optimizer = Adam(lr=1e-5)  # In[44]:
loss = 'categorical_crossentropy'  # In[45]:
metrics = ['categorical_accuracy']  # In[46]:
def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))  # In[47]:
print_layer_trainable()  # In[48]:
conv_model.trainable = False  # In[49]:
for layer in conv_model.layers:
    layer.trainable = False  # In[50]:
print_layer_trainable()  # In[51]:
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)  # In[52]:
epochs = 20
steps_per_epoch = 100  # In[53]:
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)  # In[54]:
plot_training_history(history)  # In[55]:
result = new_model.evaluate_generator(generator_test, steps=steps_test)  # In[56]:
print("Test-set classification accuracy: {0:.2%}".format(result[1]))  # In[57]:
example_errors()

# ## Fine-Tuning  # In[58]:
conv_model.trainable = True  # In[59]:
for layer in conv_model.layers:
    trainable = ('block5' in layer.name or 'block4' in layer.name)
    
    layer.trainable = trainable  # In[60]:
print_layer_trainable()  # In[61]:
optimizer_fine = Adam(lr=1e-7)  # In[62]:
new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)  # In[63]:
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)  # In[64]:
plot_training_history(history)  # In[65]:
result = new_model.evaluate_generator(generator_test, steps=steps_test)  # In[66]:
print("Test-set classification accuracy: {0:.2%}".format(result[1]))  # In[67]:
example_errors()

# ## Conclusion

# ## Exercises

# ## License (MIT)
