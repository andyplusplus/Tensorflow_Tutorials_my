# # TensorFlow Tutorial #21

# # Machine Translation

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
import math
import os  # In[2]:
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences  # In[3]:
tf.__version__  # In[4]:
tf.keras.__version__

# ## Load Data  # In[5]:
import europarl  # In[6]:
language_code='da'  # In[7]:
mark_start = 'ssss '
mark_end = ' eeee'  # In[8]:  # In[9]:
europarl.maybe_download_and_extract(language_code=language_code)  # In[10]:
data_src = europarl.load_data(english=False,
                              language_code=language_code)  # In[11]:
data_dest = europarl.load_data(english=True,
                               language_code=language_code,
                               start=mark_start,
                               end=mark_end)

# ### Example Data  # In[12]:
idx = 2  # In[13]:
data_src[idx]  # In[14]:
data_dest[idx]

# ### Error in Data  # In[15]:
idx = 8002  # In[16]:
data_src[idx]  # In[17]:
data_dest[idx]

# ## Tokenizer  # In[18]:
num_words = 10000  # In[19]:
class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""
    def __init__(self, texts, padding,
                 reverse=False, num_words=None):
        """
        :param texts: List of strings. This is the data-set.
        :param padding: Either 'post' or 'pre' padding.
        :param reverse: Boolean whether to reverse token-lists.
        :param num_words: Max number of words to use.
        """
        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))
        self.tokens = self.texts_to_sequences(texts)
        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]

            truncating = 'pre'
        else:
            truncating = 'post'
        self.num_tokens = [len(x) for x in self.tokens]
        self.max_tokens = np.mean(self.num_tokens)                           + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)
        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=self.max_tokens,
                                           padding=padding,
                                           truncating=truncating)
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
    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)
        if reverse:
            tokens = np.flip(tokens, axis=1)
            truncating = 'pre'
        else:
            truncating = 'post'
        if padding:
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)
        return tokens  # In[20]:  # In[21]:  # In[22]:
tokens_src = tokenizer_src.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded
print(tokens_src.shape)
print(tokens_dest.shape)  # In[23]:
token_start = tokenizer_dest.word_index[mark_start.strip()]
token_start  # In[24]:
token_end = tokenizer_dest.word_index[mark_end.strip()]
token_end

# ### Example of Token Sequences  # In[25]:
idx = 2  # In[26]:
tokens_src[idx]  # In[27]:
tokenizer_src.tokens_to_string(tokens_src[idx])  # In[28]:
data_src[idx]  # In[29]:
tokens_dest[idx]  # In[30]:
tokenizer_dest.tokens_to_string(tokens_dest[idx])  # In[31]:
data_dest[idx]

# ### Training Data  # In[32]:
encoder_input_data = tokens_src  # In[33]:
decoder_input_data = tokens_dest[:, :-1]
decoder_input_data.shape  # In[34]:
decoder_output_data = tokens_dest[:, 1:]
decoder_output_data.shape  # In[35]:
idx = 2  # In[36]:
decoder_input_data[idx]  # In[37]:
decoder_output_data[idx]  # In[38]:
tokenizer_dest.tokens_to_string(decoder_input_data[idx])  # In[39]:
tokenizer_dest.tokens_to_string(decoder_output_data[idx])

# ## Create the Neural Network

# ### Create the Encoder  # In[40]:
encoder_input = Input(shape=(None, ), name='encoder_input')  # In[41]:
embedding_size = 128  # In[42]:
encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')  # In[43]:
state_size = 512  # In[44]:
encoder_gru1 = GRU(state_size, name='encoder_gru1',
                   return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',
                   return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3',
                   return_sequences=False)  # In[45]:
def connect_encoder():
    net = encoder_input

    net = encoder_embedding(net)
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)
    encoder_output = net
    return encoder_output  # In[46]:
encoder_output = connect_encoder()

# ### Create the Decoder  # In[47]:
decoder_initial_state = Input(shape=(state_size,),
                              name='decoder_initial_state')  # In[48]:
decoder_input = Input(shape=(None, ), name='decoder_input')  # In[49]:
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')  # In[50]:
decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)  # In[51]:
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')  # In[52]:
def connect_decoder(initial_state):
    net = decoder_input
    net = decoder_embedding(net)

    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    decoder_output = decoder_dense(net)
    return decoder_output

# ### Connect and Create the Models  # In[53]:
decoder_output = connect_decoder(initial_state=encoder_output)
model_train = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output])  # In[54]:
model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])  # In[55]:
decoder_output = connect_decoder(initial_state=decoder_initial_state)
model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
                      outputs=[decoder_output])

# ### Loss Function  # In[56]:  # In[57]:
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

# ### Compile the Training Model  # In[58]:
optimizer = RMSprop(lr=1e-3)  # In[59]:
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))  # In[60]:
model_train.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])

# ### Callback Functions  # In[61]:
path_checkpoint = '21_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)  # In[62]:
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)  # In[63]:
callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=False)  # In[64]:
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard]

# ### Load Checkpoint  # In[65]:
try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

# ### Train the Model  # In[66]:
x_data = {
    'encoder_input': encoder_input_data,
    'decoder_input': decoder_input_data
}  # In[67]:
y_data = {
    'decoder_output': decoder_output_data
}  # In[68]:
validation_split = 10000 / len(encoder_input_data)
validation_split  # In[ ]:
model_train.fit(x=x_data,
                y=y_data,
                batch_size=640,
                epochs=10,
                validation_split=validation_split,
                callbacks=callbacks)

# ## Translate Texts  # In[69]:
def translate(input_text, true_output_text=None):
    """Translate a single text-string."""
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)

    initial_state = model_encoder.predict(input_tokens)
    max_tokens = tokenizer_dest.max_tokens
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data =         {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }
        decoder_output = model_decoder.predict(x_data)
        token_onehot = decoder_output[0, count_tokens, :]

        token_int = np.argmax(token_onehot)
        sampled_word = tokenizer_dest.token_to_word(token_int)
        output_text += " " + sampled_word
        count_tokens += 1
    output_tokens = decoder_input_data[0]

    print("Input text:")
    print(input_text)
    print()
    print("Translated text:")
    print(output_text)
    print()
    if true_output_text is not None:
        print("True output text:")
        print(true_output_text)
        print()

# ### Examples  # In[70]:
idx = 3
translate(input_text=data_src[idx],
          true_output_text=data_dest[idx])  # In[71]:
idx = 4
translate(input_text=data_src[idx],
          true_output_text=data_dest[idx])  # In[72]:
idx = 3
translate(input_text=data_src[idx] + data_src[idx+1],
          true_output_text=data_dest[idx] + data_dest[idx+1])  # In[73]:
idx = 3
translate(input_text=data_src[idx+1] + data_src[idx],
          true_output_text=data_dest[idx+1] + data_dest[idx])  # In[74]:
translate(input_text="der var engang et land der hed Danmark",
          true_output_text='Once there was a country named Denmark')  # In[75]:
translate(input_text="Idag kan man læse i avisen at Danmark er blevet fornuftigt",
          true_output_text="Today you can read in the newspaper that Denmark has become sensible.")  # In[76]:
translate(input_text="Hvem spæner ud af en butik og tygger de stærkeste bolcher?",
          true_output_text="Who runs out of a shop and chews the strongest bon-bons?")

# ## Conclusion

# ## Exercises

# ## License (MIT)
