# # TensorFlow Tutorial #23

# # Time-Series Prediction

# ## Introduction

# ## Location

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
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler  # In[2]:
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau  # In[3]:
tf.__version__  # In[4]:
tf.keras.__version__  # In[5]:
pd.__version__

# ## Load Data  # In[6]:
import weather  # In[7]:
weather.maybe_download_and_extract()  # In[8]:
cities = weather.cities
cities  # In[9]:
df = weather.load_resampled_data()  # In[10]:
df.head()

# ### Missing Data  # In[11]:
df['Esbjerg']['Pressure'].plot()  # In[12]:
df['Roskilde']['Pressure'].plot()  # In[13]:
df.values.shape  # In[14]:
df.drop(('Esbjerg', 'Pressure'), axis=1, inplace=True)
df.drop(('Roskilde', 'Pressure'), axis=1, inplace=True)  # In[15]:
df.values.shape  # In[16]:
df.head(1)

# ### Data Errors  # In[17]:
df['Odense']['Temp']['2006-05':'2006-07'].plot()  # In[18]:
df['Aarhus']['Temp']['2006-05':'2006-07'].plot()  # In[19]:
df['Roskilde']['Temp']['2006-05':'2006-07'].plot()

# ### Add Data  # In[20]:
df['Various', 'Day'] = df.index.dayofyear
df['Various', 'Hour'] = df.index.hour

# ### Target Data for Prediction  # In[21]:
target_city = 'Odense'  # In[22]:
target_names = ['Temp', 'WindSpeed', 'Pressure']  # In[23]:
shift_days = 1
shift_steps = shift_days * 24  # Number of hours.  # In[24]:
df_targets = df[target_city][target_names].shift(-shift_steps)  # In[25]:
df[target_city][target_names].head(shift_steps + 5)  # In[26]:
df_targets.head(5)  # In[27]:
df_targets.tail()

# ### NumPy Arrays  # In[28]:
x_data = df.values[0:-shift_steps]  # In[29]:
print(type(x_data))
print("Shape:", x_data.shape)  # In[30]:
y_data = df_targets.values[:-shift_steps]  # In[31]:
print(type(y_data))
print("Shape:", y_data.shape)  # In[32]:
num_data = len(x_data)
num_data  # In[33]:
train_split = 0.9  # In[34]:
num_train = int(train_split * num_data)
num_train  # In[35]:
num_test = num_data - num_train
num_test  # In[36]:
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)  # In[37]:
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)  # In[38]:
num_x_signals = x_data.shape[1]
num_x_signals  # In[39]:
num_y_signals = y_data.shape[1]
num_y_signals

# ### Scaled Data  # In[40]:
print("Min:", np.min(x_train))
print("Max:", np.max(x_train))  # In[41]:
x_scaler = MinMaxScaler()  # In[42]:
x_train_scaled = x_scaler.fit_transform(x_train)  # In[43]:
print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))  # In[44]:
x_test_scaled = x_scaler.transform(x_test)  # In[45]:
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# ## Data Generator  # In[46]:
print(x_train_scaled.shape)
print(y_train_scaled.shape)  # In[47]:
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """
    while True:
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        for i in range(batch_size):
            idx = np.random.randint(num_train - sequence_length)

            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        yield (x_batch, y_batch)  # In[48]:
batch_size = 256  # In[49]:
sequence_length = 24 * 7 * 8
sequence_length  # In[50]:
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)  # In[51]:
x_batch, y_batch = next(generator)  # In[52]:
print(x_batch.shape)
print(y_batch.shape)  # In[53]:
batch = 0   # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
plt.plot(seq)  # In[54]:
seq = y_batch[batch, :, signal]
plt.plot(seq)

# ### Validation Set  # In[55]:
validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

# ## Create the Recurrent Neural Network  # In[56]:
model = Sequential()  # In[57]:
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))  # In[58]:
model.add(Dense(num_y_signals, activation='sigmoid'))  # In[59]:
if False:
    from tensorflow.python.keras.initializers import RandomUniform
    init = RandomUniform(minval=-0.05, maxval=0.05)
    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))

# ### Loss Function  # In[60]:
warmup_steps = 50  # In[61]:
def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    y_true is the desired output.
    y_pred is the model's output.
    """
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

# ### Compile Model  # In[62]:
optimizer = RMSprop(lr=1e-3)  # In[63]:
model.compile(loss=loss_mse_warmup, optimizer=optimizer)  # In[64]:
model.summary()

# ### Callback Functions  # In[65]:
path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)  # In[66]:
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)  # In[67]:
callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)  # In[68]:
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)  # In[69]:
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

# ## Train the Recurrent Neural Network  # In[ ]:

# ### Load Checkpoint  # In[70]:
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

# ## Performance on Test-Set  # In[71]:
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))  # In[72]:
print("loss (test-set):", result)  # In[73]:
if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))

# ## Generate Predictions  # In[74]:
def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    if train:
        x = x_train_scaled
        y_true = y_train
    else:
        x = x_test_scaled
        y_true = y_test

    end_idx = start_idx + length

    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)

    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    for signal in range(len(target_names)):
        signal_pred = y_pred_rescaled[:, signal]

        signal_true = y_true[:, signal]
        plt.figure(figsize=(15,5))

        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')

        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        plt.ylabel(target_names[signal])
        plt.legend()
        if is_plot: plt.show()  # In[75]:
plot_comparison(start_idx=100000, length=1000, train=True)

# ### Strange Example  # In[76]:
plot_comparison(start_idx=200000, length=1000, train=True)  # In[77]:
df['Odense']['Temp'][200000:200000+1000].plot()  # In[78]:
df_org = weather.load_original_data()
df_org.xs('Odense')['Temp']['2002-12-23':'2003-02-04'].plot()

# ### Example from Test-Set  # In[79]:
plot_comparison(start_idx=200, length=1000, train=False)

# ## Conclusion

# ## Exercises

# ## License (MIT)
