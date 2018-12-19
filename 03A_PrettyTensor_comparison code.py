

# In[20]: # ### Graph Construction

if False:  # Don't execute this! Just show it for easy comparison.

    # In[17]: # The following helper-function creates a new convolutional network. The input and output are 4-dimensional tensors (aka. 4-rank tensors). Note the low-level details of the TensorFlow API, such as the shape of the weights-variable. It is easy to make a mistake somewhere which may result in strange error-messages that are difficult to debug.

    # ## TensorFlow Implementation
    # In[15]: # ### Helper-functions
    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))


    # First convolutional layer.
    def new_conv_layer(input,              # The previous layer.
                       num_input_channels, # Num. channels in prev. layer.
                       filter_size,        # Width and height of filters.
                       num_filters,        # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.

        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = new_weights(shape=shape)
        biases = new_biases(length=num_filters)
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        layer += biases

        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
        layer = tf.nn.relu(layer)
        return layer, weights


    # In[18]: # The following helper-function flattens a 4-dim tensor to 2-dim so we can add fully-connected layers after the convolutional layers.
    def flatten_layer(layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features


    # In[19]: # The following helper-function creates a fully-connected layer.
    def new_fc_layer(input,          # The previous layer.
                     num_inputs,     # Num. inputs from prev. layer.
                     num_outputs,    # Num. outputs.
                     use_relu=True): # Use Rectified Linear Unit (ReLU)?
        weights = new_weights(shape=[num_inputs, num_outputs])
        biases = new_biases(length=num_outputs)
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer



    layer_conv1, weights_conv1 =         new_conv_layer(input=x_image,
                       num_input_channels=num_channels,
                       filter_size=5,
                       num_filters=16,
                       use_pooling=True)

    # Second convolutional layer.
    layer_conv2, weights_conv2 =         new_conv_layer(input=layer_conv1,
                       num_input_channels=16,
                       filter_size=5,
                       num_filters=36,
                       use_pooling=True)

    # Flatten layer.
    layer_flat, num_features = flatten_layer(layer_conv2)

    # First fully-connected layer.
    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=128,
                             use_relu=True)

    # Second fully-connected layer.
    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=128,
                             num_outputs=num_classes,
                             use_relu=False)

    # Predicted class-label.
    y_pred = tf.nn.softmax(layer_fc2)

    # Cross-entropy for the classification of each image.
    cross_entropy =         tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                labels=y_true)

    # Loss aka. cost-measure. # This is the scalar value that must be minimized.
    loss = tf.reduce_mean(cross_entropy)



