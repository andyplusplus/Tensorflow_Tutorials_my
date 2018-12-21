# # TensorFlow Tutorial #18

# # TFRecords & Dataset API

# ## Introduction

# ## Imports  # In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf
from common.time_usage import get_start_time
from common.time_usage import print_time_usage
start_time_global=get_start_time()
is_plot = False
import numpy as np
import sys
import os  # In[2]:
tf.__version__

# ## Load Data  # In[3]:
import knifey  # In[4]:
from knifey import img_size, img_size_flat, img_shape, num_classes, num_channels  # In[5]:  # In[6]:
knifey.maybe_download_and_extract()  # In[7]:
dataset = knifey.load()  # In[8]:
class_names = dataset.class_names
class_names

# ### Training and Test-Sets  # In[9]:
image_paths_train, cls_train, labels_train = dataset.get_training_set()  # In[10]:
image_paths_train[0]  # In[11]:
image_paths_test, cls_test, labels_test = dataset.get_test_set()  # In[12]:
image_paths_test[0]  # In[13]:
print("Size of:")
print("- Training-set:\t\t{}".format(len(image_paths_train)))
print("- Test-set:\t\t{}".format(len(image_paths_test)))

# ### Helper-function for plotting images  # In[14]:
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
                xlabel = "True: {0}\nPred: {1}".format(cls_true_name,
                                                       cls_pred_name)
            ax.set_xlabel(xlabel)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    if is_plot: plt.show()

# ### Helper-function for loading images  # In[15]:
def load_images(image_paths):
    images = [imread(path) for path in image_paths]
    return np.asarray(images)

# ### Plot a few images to see if data is correct  # In[16]:
images = load_images(image_paths=image_paths_test[0:9])
cls_true = cls_test[0:9]
plot_images(images=images, cls_true=cls_true, smooth=True)

# ## Create TFRecords  # In[17]:
path_tfrecords_train = os.path.join(knifey.data_dir, "train.tfrecords")
path_tfrecords_train  # In[18]:
path_tfrecords_test = os.path.join(knifey.data_dir, "test.tfrecords")
path_tfrecords_test  # In[19]:
def print_progress(count, total):
    pct_complete = float(count) / total
    msg = "\r- Progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()  # In[20]:
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  # In[21]:
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  # In[22]:
def convert(image_paths, labels, out_path):
    
    print("Converting: " + out_path)
    
    num_images = len(image_paths)
    
    with tf.python_io.TFRecordWriter(out_path) as writer:
        
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            print_progress(count=i, total=num_images-1)
            img = imread(path)
            
            img_bytes = img.tostring()
            data =                 {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_int64(label)
                }
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()
            
            writer.write(serialized)  # In[23]:
convert(image_paths=image_paths_train,
        labels=cls_train,
        out_path=path_tfrecords_train)  # In[24]:
convert(image_paths=image_paths_test,
        labels=cls_test,
        out_path=path_tfrecords_test)

# ## Input Functions for the Estimator  # In[25]:
def parse(serialized):
    features =         {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)
    image_raw = parsed_example['image']
    image = tf.decode_raw(image_raw, tf.uint8)
    
    image = tf.cast(image, tf.float32)
    label = parsed_example['label']
    return image, label  # In[26]:
def input_fn(filenames, train, batch_size=32, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parse)
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        
        num_repeat = 1
    dataset = dataset.repeat(num_repeat)
    
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    x = {'image': images_batch}
    y = labels_batch
    return x, y  # In[27]:
def train_input_fn():
    return input_fn(filenames=path_tfrecords_train, train=True)  # In[28]:
def test_input_fn():
    return input_fn(filenames=path_tfrecords_test, train=False)

# ### Input Function for Predicting on New Images  # In[29]:
some_images = load_images(image_paths=image_paths_test[0:9])  # In[30]:
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": some_images.astype(np.float32)},
    num_epochs=1,
    shuffle=False)  # In[31]:
some_images_cls = cls_test[0:9]

# ## Pre-Made / Canned Estimator  # In[32]:
feature_image = tf.feature_column.numeric_column("image",
                                                 shape=img_shape)  # In[33]:
feature_columns = [feature_image]  # In[34]:
num_hidden_units = [512, 256, 128]  # In[35]:
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                   hidden_units=num_hidden_units,
                                   activation_fn=tf.nn.relu,
                                   n_classes=num_classes,
                                   model_dir="./checkpoints_tutorial18-1/")

# ### Training  # In[36]:
model.train(input_fn=train_input_fn, steps=200)

# ### Evaluation  # In[37]:
result = model.evaluate(input_fn=test_input_fn)  # In[38]:
result  # In[39]:
print("Classification accuracy: {0:.2%}".format(result["accuracy"]))

# ### Predictions  # In[40]:
predictions = model.predict(input_fn=predict_input_fn)  # In[41]:
cls = [p['classes'] for p in predictions]  # In[42]:
cls_pred = np.array(cls, dtype='int').squeeze()
cls_pred  # In[43]:
plot_images(images=some_images,
            cls_true=some_images_cls,
            cls_pred=cls_pred)

# ### Predictions for the Entire Test-Set  # In[44]:
predictions = model.predict(input_fn=test_input_fn)  # In[45]:
cls = [p['classes'] for p in predictions]  # In[46]:
cls_pred = np.array(cls, dtype='int').squeeze()  # In[47]:
np.sum(cls_pred == 2)

# # New Estimator  # In[48]:
def model_fn(features, labels, mode, params):
    
    x = features["image"]
    net = tf.reshape(x, [-1, img_size, img_size, num_channels])    
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)    
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)    
    net = tf.layers.dense(inputs=net, name='layer_fc_2',
                          units=num_classes)
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
        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec

# ### Create an Instance of the Estimator  # In[49]:
params = {"learning_rate": 1e-4}  # In[50]:
model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="./checkpoints_tutorial18-2/")

# ### Training  # In[51]:
model.train(input_fn=train_input_fn, steps=200)

# ### Evaluation  # In[52]:
result = model.evaluate(input_fn=test_input_fn)  # In[53]:
result  # In[54]:
print("Classification accuracy: {0:.2%}".format(result["accuracy"]))

# ### Predictions  # In[55]:
predictions = model.predict(input_fn=predict_input_fn)  # In[56]:
cls_pred = np.array(list(predictions))
cls_pred  # In[57]:
plot_images(images=some_images,
            cls_true=some_images_cls,
            cls_pred=cls_pred)

# ### Predictions for the Entire Test-Set  # In[58]:
predictions = model.predict(input_fn=test_input_fn)  # In[59]:
cls_pred = np.array(list(predictions))
cls_pred  # In[60]:
np.sum(cls_pred == 0)  # In[61]:
np.sum(cls_pred == 1)  # In[62]:
np.sum(cls_pred == 2)

# ## Conclusion

# ## Exercises

# ## License (MIT)
