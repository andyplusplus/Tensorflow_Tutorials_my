# # TensorFlow Tutorial #20

# # Natural Language Processing

# ## Introduction

# ## Flowchart

# ## Recurrent Neural Network

# ### Unrolled Network

# ### 3-Layer Unrolled Network

# ### Exploding & Vanishing Gradients

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
from scipy.spatial.distance import cdist  # In[2]:
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences  # In[3]:
tf.__version__  # In[4]:
tf.keras.__version__

# ## Load Data  # In[5]:
import imdb  # In[6]:  # In[7]:
imdb.maybe_download_and_extract()  # In[8]:
x_train_text, y_train = imdb.load_data(train=True)
x_test_text, y_test = imdb.load_data(train=False)  # In[9]:
print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))  # In[10]:
data_text = x_train_text + x_test_text  # In[11]:
x_train_text[1]  # In[12]:
y_train[1]

# ## Tokenizer  # In[13]:
num_words = 10000  # In[14]:
tokenizer = Tokenizer(num_words=num_words)  # In[15]:  # In[16]:
if num_words is None:
    num_words = len(tokenizer.word_index)  # In[17]:
tokenizer.word_index  # In[18]:
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)  # In[19]:
x_train_text[1]  # In[20]:
np.array(x_train_tokens[1])  # In[21]:
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

# ## Padding and Truncating Data  # In[22]:
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)  # In[23]:
np.mean(num_tokens)  # In[24]:
np.max(num_tokens)  # In[25]:
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens  # In[26]:
np.sum(num_tokens < max_tokens) / len(num_tokens)  # In[27]:
pad = 'pre'  # In[28]:
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)  # In[29]:
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)  # In[30]:
x_train_pad.shape  # In[31]:
x_test_pad.shape  # In[32]:
np.array(x_train_tokens[1])  # In[33]:
x_train_pad[1]

# ## Tokenizer Inverse Map  # In[34]:
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))  # In[35]:
def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token != 0]

    text = " ".join(words)
    return text  # In[36]:
x_train_text[1]  # In[37]:
tokens_to_string(x_train_tokens[1])

# ## Create the Recurrent Neural Network  # In[38]:
model = Sequential()  # In[39]:
embedding_size = 8  # In[40]:
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))  # In[41]:
model.add(GRU(units=16, return_sequences=True))  # In[42]:
model.add(GRU(units=8, return_sequences=True))  # In[43]:
model.add(GRU(units=4))  # In[44]:
model.add(Dense(1, activation='sigmoid'))  # In[45]:
optimizer = Adam(lr=1e-3)  # In[46]:
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])  # In[47]:
model.summary()

# ## Train the Recurrent Neural Network  # In[48]:

# ## Performance on Test-Set  # In[49]:  # In[50]:
print("Accuracy: {0:.2%}".format(result[1]))

# ## Example of Mis-Classified Text  # In[51]:  # In[52]:
cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])  # In[53]:
cls_true = np.array(y_test[0:1000])  # In[54]:
incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]  # In[55]:
len(incorrect)  # In[56]:
idx = incorrect[0]
idx  # In[57]:
text = x_test_text[idx]
text  # In[58]:
y_pred[idx]  # In[59]:
cls_true[idx]

# ## New Data  # In[60]:
text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
texts = [text1, text2, text3, text4, text5, text6, text7, text8]  # In[61]:
tokens = tokenizer.texts_to_sequences(texts)  # In[62]:
tokens_pad = pad_sequences(tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
tokens_pad.shape  # In[63]:
model.predict(tokens_pad)

# ## Embeddings  # In[64]:
layer_embedding = model.get_layer('layer_embedding')  # In[65]:
weights_embedding = layer_embedding.get_weights()[0]  # In[66]:
weights_embedding.shape  # In[67]:
token_good = tokenizer.word_index['good']
token_good  # In[68]:
token_great = tokenizer.word_index['great']
token_great  # In[69]:
weights_embedding[token_good]  # In[70]:
weights_embedding[token_great]  # In[71]:
token_bad = tokenizer.word_index['bad']
token_horrible = tokenizer.word_index['horrible']  # In[72]:
weights_embedding[token_bad]  # In[73]:
weights_embedding[token_horrible]

# ### Sorted Words  # In[74]:
def print_sorted_words(word, metric='cosine'):
    """
    Print the words in the vocabulary sorted according to their
    embedding-distance to the given word.
    Different metrics can be used, e.g. 'cosine' or 'euclidean'.
    """
    token = tokenizer.word_index[word]
    embedding = weights_embedding[token]
    distances = cdist(weights_embedding, [embedding],
                      metric=metric).T[0]

    sorted_index = np.argsort(distances)

    sorted_distances = distances[sorted_index]

    sorted_words = [inverse_map[token] for token in sorted_index
                    if token != 0]
    def _print_words(words, distances):
        for word, distance in zip(words, distances):
            print("{0:.3f} - {1}".format(distance, word))
    k = 10
    print("Distance from '{0}':".format(word))
    _print_words(sorted_words[0:k], sorted_distances[0:k])
    print("...")
    _print_words(sorted_words[-k:], sorted_distances[-k:])  # In[75]:
print_sorted_words('great', metric='cosine')  # In[76]:
print_sorted_words('worst', metric='cosine')

# ## Conclusion

# ## Exercises

# ## License (MIT)
