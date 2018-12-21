# # TensorFlow Tutorial #22

# # Image Captioning

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
import os
from PIL import Image
from cache import cache  # In[2]:
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences  # In[3]:
tf.__version__  # In[4]:
tf.keras.__version__

# ## Load Data  # In[5]:
import coco  # In[6]:  # In[7]:
coco.maybe_download_and_extract()  # In[8]:
_, filenames_train, captions_train = coco.load_records(train=True)  # In[9]:
num_images_train = len(filenames_train)
num_images_train  # In[10]:
_, filenames_val, captions_val = coco.load_records(train=False)

# ### Helper-Functions for Loading and Showing Images  # In[11]:
def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it
    to the given size if not None.
    """
    img = Image.open(path)
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)
    img = np.array(img)
    img = img / 255.0
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img  # In[12]:
def show_image(idx, train):
    """
    Load and plot an image from the training- or validation-set
    with the given index.
    """
    if train:
        dir = coco.train_dir
        filename = filenames_train[idx]
        captions = captions_train[idx]
    else:
        dir = coco.val_dir
        filename = filenames_val[idx]
        captions = captions_val[idx]
    path = os.path.join(dir, filename)
    for caption in captions:
        print(caption)

    img = load_image(path)
    plt.imshow(img)
    if is_plot: plt.show()

# ### Example Image  # In[13]:
show_image(idx=1, train=True)

# ## Pre-Trained Image Model (VGG16)  # In[14]:
image_model = VGG16(include_top=True, weights='imagenet')  # In[15]:
image_model.summary()  # In[16]:
transfer_layer = image_model.get_layer('fc2')  # In[17]:
image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)  # In[18]:
img_size = K.int_shape(image_model.input)[1:3]
img_size  # In[19]:
transfer_values_size = K.int_shape(transfer_layer.output)[1]
transfer_values_size

# ### Process All Images  # In[20]:
def print_progress(count, max_count):
    pct_complete = count / max_count
    msg = "\r- Progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()  # In[21]:
def process_images(data_dir, filenames, batch_size=32):
    """
    Process all the given files in the given data_dir using the
    pre-trained image-model and return their transfer-values.
    Note that we process the images in batches to save
    memory and improve efficiency on the GPU.
    """

    num_images = len(filenames)
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)
    start_index = 0
    while start_index < num_images:
        print_progress(count=start_index, max_count=num_images)
        end_index = start_index + batch_size
        if end_index > num_images:
            end_index = num_images
        current_batch_size = end_index - start_index
        for i, filename in enumerate(filenames[start_index:end_index]):
            path = os.path.join(data_dir, filename)
            img = load_image(path, size=img_size)
            image_batch[i] = img
        transfer_values_batch =             image_model_transfer.predict(image_batch[0:current_batch_size])
        transfer_values[start_index:end_index] =             transfer_values_batch[0:current_batch_size]
        start_index = end_index
    print()
    return transfer_values  # In[22]:
def process_images_train():
    print("Processing {0} images in training-set ...".format(len(filenames_train)))
    cache_path = os.path.join(coco.data_dir,
                              "transfer_values_train.pkl")
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir=coco.train_dir,
                            filenames=filenames_train)
    return transfer_values  # In[23]:
def process_images_val():
    print("Processing {0} images in validation-set ...".format(len(filenames_val)))
    cache_path = os.path.join(coco.data_dir, "transfer_values_val.pkl")
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir=coco.val_dir,
                            filenames=filenames_val)
    return transfer_values  # In[24]:  # In[25]:

# ## Tokenizer  # In[26]:
mark_start = 'ssss '
mark_end = ' eeee'  # In[27]:
def mark_captions(captions_listlist):
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]
    return captions_marked  # In[28]:
captions_train_marked = mark_captions(captions_train)
captions_train_marked[0]  # In[29]:
captions_train[0]  # In[30]:
def flatten(captions_listlist):
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]
    return captions_list  # In[31]:
captions_train_flat = flatten(captions_train_marked)  # In[32]:
num_words = 10000  # In[33]:
class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """
        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))
    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""
        word = " " if token == 0 else self.index_to_word[token]
        return word 
    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        text = " ".join(words)
        return text
    def captions_to_tokens(self, captions_listlist):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """

        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        return tokens  # In[34]:  # In[35]:
token_start = tokenizer.word_index[mark_start.strip()]
token_start  # In[36]:
token_end = tokenizer.word_index[mark_end.strip()]
token_end  # In[37]:  # In[38]:
tokens_train[0]  # In[39]:
captions_train_marked[0]

# ## Data Generator  # In[40]:
def get_random_caption_tokens(idx):
    """
    Given a list of indices for images in the training-set,
    select a token-sequence for a random caption,
    and return a list of all these token-sequences.
    """

    result = []
    for i in idx:
        j = np.random.choice(len(tokens_train[i]))
        tokens = tokens_train[i][j]
        result.append(tokens)
    return result  # In[41]:
def batch_generator(batch_size):
    """
    Generator function for creating random batches of training-data.
    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """
    while True:
        idx = np.random.randint(num_images_train,
                                size=batch_size)

        transfer_values = transfer_values_train[idx]
        tokens = get_random_caption_tokens(idx)
        num_tokens = [len(t) for t in tokens]

        max_tokens = np.max(num_tokens)

        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')

        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]
        x_data =         {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values
        }
        y_data =         {
            'decoder_output': decoder_output_data
        }
        yield (x_data, y_data)  # In[42]:
batch_size = 1024  # In[43]:
generator = batch_generator(batch_size=batch_size)  # In[44]:
batch = next(generator)
batch_x = batch[0]
batch_y = batch[1]  # In[45]:
batch_x['transfer_values_input'][0]  # In[46]:
batch_x['decoder_input'][0]  # In[47]:
batch_y['decoder_output'][0]

# ### Steps Per Epoch  # In[48]:
num_captions_train = [len(captions) for captions in captions_train]  # In[49]:
total_num_captions_train = np.sum(num_captions_train)  # In[50]:
steps_per_epoch = int(total_num_captions_train / batch_size)
steps_per_epoch

# ## Create the Recurrent Neural Network  # In[51]:
state_size = 512  # In[52]:
embedding_size = 128  # In[53]:
transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')  # In[54]:
decoder_transfer_map = Dense(state_size,
                             activation='tanh',
                             name='decoder_transfer_map')  # In[55]:
decoder_input = Input(shape=(None, ), name='decoder_input')  # In[56]:
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')  # In[57]:
decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)  # In[58]:
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

# ### Connect and Create the Training Model  # In[59]:
def connect_decoder(transfer_values):
    initial_state = decoder_transfer_map(transfer_values)
    net = decoder_input

    net = decoder_embedding(net)

    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    decoder_output = decoder_dense(net)
    return decoder_output  # In[60]:
decoder_output = connect_decoder(transfer_values=transfer_values_input)
decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

# ### Loss Function  # In[61]:  # In[62]:
def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.
    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.
    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

# ### Compile the Training Model  # In[63]:
optimizer = RMSprop(lr=1e-3)  # In[64]:
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))  # In[65]:
decoder_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy,
                      target_tensors=[decoder_target])

# ### Callback Functions  # In[66]:
path_checkpoint = '22_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      verbose=1,
                                      save_weights_only=True)  # In[67]:
callback_tensorboard = TensorBoard(log_dir='./22_logs/',
                                   histogram_freq=0,
                                   write_graph=False)  # In[68]:
callbacks = [callback_checkpoint, callback_tensorboard]

# ### Load Checkpoint  # In[69]:
try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

# ### Train the Model  # In[ ]:

# ## Generate Captions  # In[70]:
def generate_caption(image_path, max_tokens=30):
    """
    Generate a caption for the image in the given path.
    The caption is limited to the given number of tokens (words).
    """
    image = load_image(image_path, size=img_size)

    image_batch = np.expand_dims(image, axis=0)
    transfer_values = image_model_transfer.predict(image_batch)
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data =         {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }
        
        decoder_output = decoder_model.predict(x_data)
        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        sampled_word = tokenizer.token_to_word(token_int)
        output_text += " " + sampled_word
        count_tokens += 1
    output_tokens = decoder_input_data[0]
    plt.imshow(image)
    if is_plot: plt.show()

    print("Predicted caption:")
    print(output_text)
    print()

# ### Examples  # In[71]:
generate_caption("data/images/parrot_cropped1.jpg")  # In[72]:
generate_caption("data/images/elon_musk.jpg")  # In[73]:
def generate_caption_coco(idx, train=False):
    """
    Generate a caption for an image in the COCO data-set.
    Use the image with the given index in either the
    training-set (train=True) or validation-set (train=False).
    """
    if train:
        data_dir = coco.train_dir
        filename = filenames_train[idx]
        captions = captions_train[idx]
    else:
        data_dir = coco.val_dir
        filename = filenames_val[idx]
        captions = captions_val[idx]
    path = os.path.join(data_dir, filename)
    generate_caption(image_path=path)
    print("True captions:")
    for caption in captions:
        print(caption)  # In[74]:
generate_caption_coco(idx=1, train=True)  # In[75]:
generate_caption_coco(idx=10, train=True)  # In[76]:
generate_caption_coco(idx=1, train=False)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

# ## Conclusion

# ## Exercises

# ## License (MIT)
