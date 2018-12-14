# # TensorFlow Tutorial #21
# # Machine Translation
# / [GitHub](https://github.com/Hvass-Labs/TensorFlow-Tutorials) / [Videos on YouTube](https://www.youtube.com/playlist?list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ)

# ## Introduction
# Tutorial #20 showed how to use a Recurrent Neural Network (RNN) to do so-called sentiment analysis on texts of movie reviews. This tutorial will extend that idea to do Machine Translation of human languages by combining two RNN's.
# You should be familiar with TensorFlow, Keras and the basics of Natural Language Processing, see Tutorials #01, #03-C and #20.

# ## Flowchart
# The following flowchart shows roughly how the neural network is constructed. It is split into two parts: An encoder which maps the source-text to a "thought vector" that summarizes the text's contents, which is then input to the second part of the neural network that decodes the "thought vector" to the destination-text.
# The neural network cannot work directly on text so first we need to convert each word to an integer-token using a tokenizer. But the neural network cannot work on integers either, so we use a so-called Embedding Layer to convert each integer-token to a vector of floating-point values. The embedding is trained alongside the rest of the neural network to map words with similar semantic meaning to similar vectors of floating-point values.
# For example, consider the Danish text "der var engang" which is the beginning of any fairytale and literally means "there was once" but is commonly translated into English as "once upon a time". We first convert the entire data-set to integer-tokens so the text "der var engang" becomes [12, 54, 1097]. Each of these integer-tokens is then mapped to an embedding-vector with e.g. 128 elements, so the integer-token 12 could for example become [0.12, -0.56, ..., 1.19] and the integer-token 54 could for example become [0.39, 0.09, ..., -0.12]. These embedding-vectors can then be input to the Recurrent Neural Network, which has 3 GRU-layers. See Tutorial #20 for a more detailed explanation.
# The last GRU-layer outputs a single vector - the "thought vector" that summarizes the contents of the source-text - which is then used as the initial state of the GRU-units in the decoder-part.
# The destination-text "once upon a time" is padded with special markers "ssss" and "eeee" to indicate its beginning and end, so the sequence of integer-tokens becomes [2, 337, 640, 9, 79, 3]. During training, the decoder will be given this entire sequence as input and  the desired output sequence is [337, 640, 9, 79, 3] which is the same sequence but time-shifted one step. We are trying to teach the decoder to map the "thought vector" and the start-token "ssss" (integer 2) to the next word "once" (integer 337), and then map the word "once" to the word "upon" (integer 640), and so forth.
# This flow-chart depicts the main idea but does not show all the necessary details e.g. regarding the loss function which is also somewhat complicated.

# ![Flowchart](images/21_machine_translation_flowchart.png)

# ## Imports

# In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os

# We need to import several things from Keras.

# In[2]:
# from tf.keras.models import Model  # This does not work!
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# This was developed using Python 3.6 (Anaconda) and package versions:

# In[3]:
tf.__version__

# In[4]:
tf.keras.__version__

# ## Load Data
# We will use the Europarl data-set which has sentence-pairs in most European languages. The data was created by the European Union which translates a lot of their communications to the languages of the member-countries of the European Union.
# http://www.statmt.org/europarl/

# In[5]:
import europarl

# In this tutorial I have used the English-Danish data-set which contains about 2 million sentence-pairs. You can use another language by changing this language-code, see `europarl.py` for a list of available language-codes.

# In[6]:
language_code='da'

# In order for the decoder to know when to begin and end a sentence, we need to mark the start and end of each sentence with words that most likely don't occur in the data-set.

# In[7]:
mark_start = 'ssss '
mark_end = ' eeee'

# You can change the directory for the data-files if you like.

# In[8]:
# data_dir = "data/europarl/"

# This will automatically download and extract the data-files if you don't have them already.
# **WARNING: The file for the English-Danish data-set is about 587 MB!**

# In[9]:
europarl.maybe_download_and_extract(language_code=language_code)

# Load the texts for the source-language, here we use Danish.

# In[10]:
data_src = europarl.load_data(english=False,
                              language_code=language_code)

# Load the texts for the destination-language, here we use English.

# In[11]:
data_dest = europarl.load_data(english=True,
                               language_code=language_code,
                               start=mark_start,
                               end=mark_end)

# We will build a model to translate from the source language (Danish) to the destination language (English). If you want to make the inverse translation you can merely exchange the source and destination data.

# ### Example Data
# The data is just a list of texts that is ordered so the source and destination texts match. I can confirm that this example is an accurate translation.

# In[12]:
idx = 2

# In[13]:
data_src[idx]

# In[14]:
data_dest[idx]

# ### Error in Data
# The data-set contains about 2 million sentence-pairs. Some of the data is incorrect. This example appears to be French (or some other weird language I don't understand), although the Danish text is also included.

# In[15]:
idx = 8002

# In[16]:
data_src[idx]

# In[17]:
data_dest[idx]

# ## Tokenizer
# Neural Networks cannot work directly on text-data. We use a two-step process to convert text into numbers that can be used in a neural network. The first step is to convert text-words into so-called integer-tokens. The second step is to convert integer-tokens into vectors of floating-point numbers using a so-called embedding-layer. See Tutorial #20 for a more detailed explanation.
# Set the maximum number of words in our vocabulary. This means that we will only use e.g. the 10000 most frequent words in the data-set. We use the same number for both the source and destination languages, but these could be different.

# In[18]:
num_words = 10000

# We need a few more functions than provided by Keras' Tokenizer-class so we wrap it.

# In[19]:
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

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

        # Convert all texts to lists of integer-tokens.
        # Note that the sequences may have different lengths.
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            # Reverse the token-sequences.
            self.tokens = [list(reversed(x)) for x in self.tokens]
        
            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        # The number of integer-tokens in each sequence.
        self.num_tokens = [len(x) for x in self.tokens]

        # Max number of tokens to use in all sequences.
        # We will pad / truncate all sequences to this length.
        # This is a compromise so we save a lot of memory and
        # only have to truncate maybe 5% of all the sequences.
        self.max_tokens = np.mean(self.num_tokens)                           + 2 * np.std(self.num_tokens)
        self.max_tokens = int(self.max_tokens)

        # Pad / truncate all token-sequences to the given length.
        # This creates a 2-dim numpy matrix that is easier to use.
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

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        
        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text
    
    def text_to_tokens(self, text, reverse=False, padding=False):
        """
        Convert a single text-string to tokens with optional
        reversal and padding.
        """

        # Convert to tokens. Note that we assume there is only
        # a single text-string so we wrap it in a list.
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_tokens,
                                   padding='pre',
                                   truncating=truncating)

        return tokens

# Now create a tokenizer for the source-language. Note that we pad zeros at the beginning ('pre') of the sequences. We also reverse the sequences of tokens because the research literature suggests that this might improve performance, because the last words seen by the encoder match the first words produced by the decoder, so short-term dependencies are supposedly modelled more accurately.

# In[20]:
get_ipython().run_cell_magic('time', '', "tokenizer_src = TokenizerWrap(texts=data_src,\n                              padding='pre',\n                              reverse=True,\n                              num_words=num_words)")

# Now create the tokenizer for the destination language. We need a tokenizer for both the source- and destination-languages because their vocabularies are different. Note that this tokenizer does not reverse the sequences and it pads zeros at the end ('post') of the arrays.

# In[21]:
get_ipython().run_cell_magic('time', '', "tokenizer_dest = TokenizerWrap(texts=data_dest,\n                               padding='post',\n                               reverse=False,\n                               num_words=num_words)")

# Define convenience variables for the padded token sequences. These are just 2-dimensional numpy arrays of integer-tokens.
# Note that the sequence-lengths are different for the source and destination languages. This is because texts with the same meaning may have different numbers of words in the two languages. 
# Furthermore, we have made a compromise when tokenizing the original texts in order to save a lot of memory. This means we only truncate about 5% of the texts.

# In[22]:
tokens_src = tokenizer_src.tokens_padded
tokens_dest = tokenizer_dest.tokens_padded
print(tokens_src.shape)
print(tokens_dest.shape)

# This is the integer-token used to mark the beginning of a text in the destination-language.

# In[23]:
token_start = tokenizer_dest.word_index[mark_start.strip()]
token_start

# This is the integer-token used to mark the end of a text in the destination-language.

# In[24]:
token_end = tokenizer_dest.word_index[mark_end.strip()]
token_end

# ### Example of Token Sequences

# This is the output of the tokenizer. Note how it is padded with zeros at the beginning (pre-padding).

# In[25]:
idx = 2

# In[26]:
tokens_src[idx]

# We can reconstruct the original text by converting each integer-token back to its corresponding word:

# In[27]:
tokenizer_src.tokens_to_string(tokens_src[idx])

# This text is actually reversed, as can be seen when compared to the original text from the data-set:

# In[28]:
data_src[idx]

# This is the sequence of integer-tokens for the corresponding text in the destination-language. Note how it is padded with zeros at the end (post-padding).

# In[29]:
tokens_dest[idx]

# We can reconstruct the original text by converting each integer-token back to its corresponding word:

# In[30]:
tokenizer_dest.tokens_to_string(tokens_dest[idx])

# Compare this to the original text from the data-set, which is almost identical except for punctuation marks and a few words such as "dreaded millennium bug". This is because we only use a vocabulary of the 10000 most frequent words in the data-set and those 3 words were apparently not used frequently enough to be included in the vocabulary, so they are merely skipped.

# In[31]:
data_dest[idx]

# ### Training Data
# Now that the data-set has been converted to sequences of integer-tokens that are padded and truncated and saved in numpy arrays, we can easily prepare the data for use in training the neural network.
# The input to the encoder is merely the numpy array for the padded and truncated sequences of integer-tokens produced by the tokenizer:

# In[32]:
encoder_input_data = tokens_src

# The input and output data for the decoder is identical, except shifted one time-step. We can use the same numpy array to save memory by slicing it, which merely creates different 'views' of the same data in memory.

# In[33]:
decoder_input_data = tokens_dest[:, :-1]
decoder_input_data.shape

# In[34]:
decoder_output_data = tokens_dest[:, 1:]
decoder_output_data.shape

# For example, these token-sequences are identical except they are shifted one time-step.

# In[35]:
idx = 2

# In[36]:
decoder_input_data[idx]

# In[37]:
decoder_output_data[idx]

# If we use the tokenizer to convert these sequences back into text, we see that they are identical except for the first word which is 'ssss' that marks the beginning of a text.

# In[38]:
tokenizer_dest.tokens_to_string(decoder_input_data[idx])

# In[39]:
tokenizer_dest.tokens_to_string(decoder_output_data[idx])

# ## Create the Neural Network
# ### Create the Encoder
# First we create the encoder-part of the neural network which maps a sequence of integer-tokens to a "thought vector". We will use the so-called functional API of Keras for this, where we first create the objects for all the layers of the neural network and then we connect them later, this allows for more flexibility than the so-called sequential API in Keras, which is useful when experimenting with more complicated architectures and ways of connecting the encoder and decoder.

# This is the input for the encoder which takes batches of integer-token sequences. The `None` indicates that the sequences can have arbitrary length.

# In[40]:
encoder_input = Input(shape=(None, ), name='encoder_input')

# This is the length of the vectors output by the embedding-layer, which maps integer-tokens to vectors of values roughly between -1 and 1, so that words that have similar semantic meanings are mapped to vectors that are similar. See Tutorial #20 for a more detailed explanation of this.

# In[41]:
embedding_size = 128

# This is the embedding-layer.

# In[42]:
encoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')

# This is the size of the internal states of the Gated Recurrent Units (GRU). The same size is used in both the encoder and decoder.

# In[43]:
state_size = 512

# This creates the 3 GRU layers that will map from a sequence of embedding-vectors to a single "thought vector" which summarizes the contents of the input-text. Note that the last GRU-layer does not return a sequence.

# In[44]:
encoder_gru1 = GRU(state_size, name='encoder_gru1',
                   return_sequences=True)
encoder_gru2 = GRU(state_size, name='encoder_gru2',
                   return_sequences=True)
encoder_gru3 = GRU(state_size, name='encoder_gru3',
                   return_sequences=False)

# This helper-function connects all the layers of the encoder.

# In[45]:
def connect_encoder():
    # Start the neural network with its input-layer.
    net = encoder_input
    
    # Connect the embedding-layer.
    net = encoder_embedding(net)

    # Connect all the GRU-layers.
    net = encoder_gru1(net)
    net = encoder_gru2(net)
    net = encoder_gru3(net)

    # This is the output of the encoder.
    encoder_output = net
    
    return encoder_output

# Note how the encoder uses the normal output from its last GRU-layer as the "thought vector". Research papers often use the internal state of the encoder's last recurrent layer as the "thought vector". But this makes the implementation more complicated and is not necessary when using the GRU. But if you were using the LSTM instead then it is necessary to use the LSTM's internal states as the "thought vector" because it actually has two internal vectors, which we would need to initialize the two internal states of the decoder's LSTM units.
# We can now use this function to connect all the layers in the encoder so it can be connected to the decoder further below.

# In[46]:
encoder_output = connect_encoder()

# ### Create the Decoder
# Create the decoder-part which maps the "thought vector" to a sequence of integer-tokens.
# The decoder takes two inputs. First it needs the "thought vector" produced by the encoder which summarizes the contents of the input-text.

# In[47]:
decoder_initial_state = Input(shape=(state_size,),
                              name='decoder_initial_state')

# The decoder also needs a sequence of integer-tokens as inputs. During training we will supply this with a full sequence of integer-tokens e.g. corresponding to the text "ssss once upon a time eeee". 
# During inference when we are translating new input-texts, we will start by feeding a sequence with just one integer-token for "ssss" which marks the beginning of a text, and combined with the "thought vector" from the encoder, the decoder will hopefully be able to produce the correct next word e.g. "once".

# In[48]:
decoder_input = Input(shape=(None, ), name='decoder_input')

# This is the embedding-layer which converts integer-tokens to vectors of real-valued numbers roughly between -1 and 1. Note that we have different embedding-layers for the encoder and decoder because we have two different vocabularies and two different tokenizers for the source and destination languages.

# In[49]:
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

# This creates the 3 GRU layers of the decoder. Note that they all return sequences because we ultimately want to output a sequence of integer-tokens that can be converted into a text-sequence.

# In[50]:
decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)

# The GRU layers output a tensor with shape `[batch_size, sequence_length, state_size]`, where each "word" is encoded as a vector of length `state_size`. We need to convert this into sequences of integer-tokens that can be interpreted as words from our vocabulary.
# One way of doing this is to convert the GRU output to a one-hot encoded array. It works but it is extremely wasteful, because for a vocabulary of e.g. 10000 words we need a vector with 10000 elements, so we can select the index of the highest element to be the integer-token.
# Note that the activation-function is set to `linear` instead of `softmax` as we would normally use for one-hot encoded outputs, because there is apparently a bug in Keras so we need to make our own loss-function, as described in detail further below.

# In[51]:
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

# The decoder is built using the functional API of Keras, which allows more flexibility in connecting the layers e.g. to route different inputs to the decoder. This is useful because we have to connect the decoder directly to the encoder, but we will also connect the decoder to another input so we can run it separately.
# This function connects all the layers of the decoder to some input of the initial-state values for the GRU layers.

# In[52]:
def connect_decoder(initial_state):
    # Start the decoder-network with its input-layer.
    net = decoder_input

    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU-layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output

# ### Connect and Create the Models
# We can now connect the encoder and decoder in different ways.
# First we connect the encoder directly to the decoder so it is one whole model that can be trained end-to-end. This means the initial-state of the decoder's GRU units are set to the output of the encoder.

# In[53]:
decoder_output = connect_decoder(initial_state=encoder_output)

model_train = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output])

# Then we create a model for just the encoder alone. This is useful for mapping a sequence of integer-tokens to a "thought-vector" summarizing its contents.

# In[54]:
model_encoder = Model(inputs=[encoder_input],
                      outputs=[encoder_output])

# Then we create a model for just the decoder alone. This allows us to directly input the initial state for the decoder's GRU units.

# In[55]:
decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state],
                      outputs=[decoder_output])

# Note that all these models use the same weights and variables of the encoder and decoder. We are merely changing how they are connected. So once the entire model has been trained, we can run the encoder and decoder models separately with the trained weights.

# ### Loss Function
# The output of the decoder is a sequence of one-hot encoded arrays. In order to train the decoder we need to supply the one-hot encoded arrays that we desire to see on the decoder's output, and then use a loss-function like cross-entropy to train the decoder to produce this desired output.
# However, our data-set contains integer-tokens instead of one-hot encoded arrays. Each one-hot encoded array has 10000 elements so it would be extremely wasteful to convert the entire data-set to one-hot encoded arrays.
# A better way is to use a so-called sparse cross-entropy loss-function, which does the conversion internally from integers to one-hot encoded arrays. Unfortunately, there seems to be a bug in Keras when using this with Recurrent Neural Networks, so the following does not work:

# In[56]:
# model_train.compile(optimizer=optimizer,
#                     loss='sparse_categorical_crossentropy')

# The decoder outputs a 3-rank tensor with shape `[batch_size, sequence_length, num_words]` which contains batches of sequences of one-hot encoded arrays of length `num_words`. We will compare this to a 2-rank tensor with shape `[batch_size, sequence_length]` containing sequences of integer-tokens.
# This comparison is done with a sparse-cross-entropy function directly from TensorFlow. There are several things to note here.
# Firstly, the loss-function calculates the softmax internally to improve numerical stability - this is why we used a linear activation function in the last dense-layer of the decoder-network above.
# Secondly, the loss-function from TensorFlow will output a 2-rank tensor of shape `[batch_size, sequence_length]` given these inputs. But this must ultimately be reduced to a single scalar-value whose gradient can be derived by TensorFlow so it can be optimized using gradient descent. Keras supports some weighting of loss-values across the batch but the semantics are unclear so to be sure that we calculate the loss-function across the entire batch and across the entire sequences, we manually calculate the loss average.

# In[57]:
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

    # Calculate the loss. This outputs a
    # 2-rank tensor of shape [batch_size, sequence_length]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

# ### Compile the Training Model
# We have used the Adam optimizer in many of the previous tutorials, but it seems to diverge in some of these experiments with Recurrent Neural Networks. RMSprop seems to work much better for these.

# In[58]:
optimizer = RMSprop(lr=1e-3)

# There seems to be another bug in Keras so it cannot automatically deduce the correct shape of the decoder's output data. We therefore need to manually create a placeholder variable for the decoder's output. The shape is set to `(None, None)` which means the batch can have an arbitrary number of sequences, which can have an arbitrary number of integer-tokens.

# In[59]:
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))

# We can now compile the model using our custom loss-function.

# In[60]:
model_train.compile(optimizer=optimizer,
                    loss=sparse_cross_entropy,
                    target_tensors=[decoder_target])

# ### Callback Functions
# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
# This is the callback for writing checkpoints during training.

# In[61]:
path_checkpoint = '21_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

# This is the callback for stopping the optimization when performance worsens on the validation-set.

# In[62]:
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

# This is the callback for writing the TensorBoard log during training.

# In[63]:
callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

# In[64]:
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard]

# ### Load Checkpoint
# You can reload the last saved checkpoint so you don't have to train the model every time you want to use it.

# In[65]:
try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

# ### Train the Model
# We wrap the data in named dicts so we are sure the data is assigned correctly to the inputs and outputs of the model.

# In[66]:
x_data = {
    'encoder_input': encoder_input_data,
    'decoder_input': decoder_input_data
}

# In[67]:
y_data = {
    'decoder_output': decoder_output_data
}

# We want a validation-set of 10000 sequences but Keras needs this number as a fraction.

# In[68]:
validation_split = 10000 / len(encoder_input_data)
validation_split

# Now we can train the model. One epoch of training took about 1 hour on a GTX 1070 GPU. You probably need to run 10 epochs or more during training. After 10 epochs the loss was about 1.10 on the training-set and about 1.15 on the validation-set.
# Note the strange batch-size of 640 (512 + 128) which was chosen because it kept the GPU running at nearly 100% while being within the memory limits of 8GB for this GPU.

# In[ ]:
model_train.fit(x=x_data,
                y=y_data,
                batch_size=640,
                epochs=10,
                validation_split=validation_split,
                callbacks=callbacks)

# ## Translate Texts
# This function translates a text from the source-language to the destination-language and optionally prints a true translation.

# In[69]:
def translate(input_text, true_output_text=None):
    """Translate a single text-string."""

    # Convert the input-text to integer-tokens.
    # Note the sequence of tokens has to be reversed.
    # Padding is probably not necessary.
    input_tokens = tokenizer_src.text_to_tokens(text=input_text,
                                                reverse=True,
                                                padding=True)
    
    # Get the output of the encoder's GRU which will be
    # used as the initial state in the decoder's GRU.
    # This could also have been the encoder's final state
    # but that is really only necessary if the encoder
    # and decoder use the LSTM instead of GRU because
    # the LSTM has two internal states.
    initial_state = model_encoder.predict(input_tokens)

    # Max number of tokens / words in the output sequence.
    max_tokens = tokenizer_dest.max_tokens

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data =         {
            'decoder_initial_state': initial_state,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.

        # Input this data to the decoder and get the predicted output.
        decoder_output = model_decoder.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        token_onehot = decoder_output[0, count_tokens, :]
        
        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer_dest.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # Sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]
    
    # Print the input-text.
    print("Input text:")
    print(input_text)
    print()

    # Print the translated output-text.
    print("Translated text:")
    print(output_text)
    print()

    # Optionally print the true translated text.
    if true_output_text is not None:
        print("True output text:")
        print(true_output_text)
        print()

# ### Examples
# Translate a text from the training-data. This translation is quite good. Note how it is not identical to the translation from the training-data, but the actual meaning is similar.

# In[70]:
idx = 3
translate(input_text=data_src[idx],
          true_output_text=data_dest[idx])

# Here is another example which is also a reasonable translation, although it has incorrectly translated the natural disasters. Note "countries of the European Union" has instead been translated as "member states" which are synonyms in this context.

# In[71]:
idx = 4
translate(input_text=data_src[idx],
          true_output_text=data_dest[idx])

# In this example we join two texts from the training-set. The model first sends this combined text through the encoder, which produces a "thought-vector" that seems to summarize both texts reasonably well so the decoder can produce a reasonable translation.

# In[72]:
idx = 3
translate(input_text=data_src[idx] + data_src[idx+1],
          true_output_text=data_dest[idx] + data_dest[idx+1])

# If we reverse the order of these two texts then the meaning is not quite so clear for the latter text.

# In[73]:
idx = 3
translate(input_text=data_src[idx+1] + data_src[idx],
          true_output_text=data_dest[idx+1] + data_dest[idx])

# This is an example I made up. It is a quite broken translation.

# In[74]:
translate(input_text="der var engang et land der hed Danmark",
          true_output_text='Once there was a country named Denmark')

# This is another example I made up. This is a better translation even though it is perhaps a more complicated text.

# In[75]:
translate(input_text="Idag kan man læse i avisen at Danmark er blevet fornuftigt",
          true_output_text="Today you can read in the newspaper that Denmark has become sensible.")

# This is a text from a Danish song. It doesn't even make much sense in Danish. However the translation is probably so broken because several of the words are not in the vocabulary.

# In[76]:
translate(input_text="Hvem spæner ud af en butik og tygger de stærkeste bolcher?",
          true_output_text="Who runs out of a shop and chews the strongest bon-bons?")

# ## Conclusion
# This tutorial showed the basic idea of using two Recurrent Neural Networks in a so-called encoder/decoder model to do Machine Translation of human languages. It was demonstrated on the very large Europarl data-set from the European Union.
# The model could produce reasonable translations for some texts but not for others. It is possible that a better architecture for the neural network and more training epochs could improve performance. There are also more advanced models that are known to improve quality of the translations.
# However, it is important to note that these models do not really understand human language. The models have no knowledge of the actual meaning of the words. The models are merely very advanced function approximators that can map between sequences of integer-tokens.

# ## Exercises
# These are a few suggestions for exercises that may help improve your skills with TensorFlow. It is important to get hands-on experience with TensorFlow in order to learn how to use it properly.
# You may want to backup this Notebook before making any changes.
# * Train for more than 10 epochs. Does it improve the translations?
# * Increase the size of the vocabulary. Does it improve the translations? Would it make sense to have different sizes for the vocabularies of the source and destination languages?
# * Find another data-set and use it together with Europarl.
# * Change the architectures of the neural network, for example change the state-size for the GRU layers, the number of GRU layers, the embedding-size, etc. Does it improve the translations?
# * Use hyper-parameter optimization from Tutorial #19 to automatically find the best hyper-parameters.
# * When translating texts, instead of using `np.argmax()` to sample the next integer-token, could you sample the decoder's output as if it was a probability distribution instead? Note that the decoder's output is not softmax-limited so you have to do that first to turn it into a probability-distribution.
# * Can you generate multiple sequences by doing this sampling? Can you find a way to select the best of these different sequences?
# * Disable the reversal of words for the source-language. Does it improve the translations?
# * What is a Bi-Directional GRU and can you use it here?
# * We use the **output** of the encoder's GRU as the initial state of the decoder's GRU. The research literature often uses an LSTM instead of the GRU, so they used the encoder's **state** instead of its output as the initial state of the decoder. Can you rewrite this code to use the encoder's state as the decoder's initial state? Is there a reason to do this, or is the encoder's output sufficient to use as the decoder's initial state?
# * Is it possible to connect multiple encoders and decoders in a single neural network, so that you can train it on different languages and allow for direct translation e.g. from Danish to Polish, German and French?
# * Explain to a friend how the program works.

# ## License (MIT)
# Copyright (c) 2018 by [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org/)
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
