import tensorflow as tf

# for input image
width = 352
hight = 288
channels = 3

learning_rate =0.0002
training_epochs = 100
tf.reset_default_graph()


layer_1 = 32  # number of filter of first layer
layer_2 = 36  # number of filter of second layer
layer_3 = 48  # number of filter of third layer
layer_4 = 64  # number of filter of  fourth  layer
layer_5 = 64  # number of filter of  fifth layer
layer_6 = 128  # number of filter of sixth layer
layer_7 = 128  # number of filter of  seventh  layer
layer_8 = 256  # number of filter of  eighth layer


def model(input_layer, num_classes=3):
    with tf.variable_scope('layer_1', reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name="weights1", shape=[5, 5, 3, layer_1], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases1", shape=[layer_1], initializer=tf.zeros_initializer())
        layer_1_output = tf.nn.conv2d(input_layer, weights, strides=[1, 2, 2, 1], padding="VALID")
        layer_1_output = tf.nn.bias_add(layer_1_output, biases)
        layer_1_output = tf.nn.relu(layer_1_output)

    print("con1 = {}".format(layer_1_output.shape))


    with tf.variable_scope('layer_2',reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name = "weights2", shape=[5, 5, layer_1, layer_2], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name = "biases2", shape = [layer_2], initializer = tf.zeros_initializer())
        layer_2_output = tf.nn.conv2d(layer_1_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_2_output = tf.nn.bias_add(layer_2_output, biases)
        layer_2_output = tf.nn.relu(layer_2_output)

    print("con2 = {}".format(layer_2_output.shape))

    with tf.variable_scope('layer_3',reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name = "weights3", shape=[5, 5, layer_2, layer_3], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name = "biases3", shape = [layer_3], initializer = tf.zeros_initializer())
        layer_3_output = tf.nn.conv2d(layer_2_output, weights,strides=[1, 2, 2, 1], padding="VALID")
        layer_3_output = tf.nn.bias_add(layer_3_output, biases)
        layer_3_output = tf.nn.relu(layer_3_output)

    print("con2 = {}".format(layer_3_output.shape))


    with tf.variable_scope('layer_4',reuse=tf.AUTO_REUSE):
      weights = tf.get_variable(name="weights4", shape=[3, 3,layer_3 , layer_4], initializer=tf.contrib.layers.xavier_initializer())
      biases = tf.get_variable(name="biases4", shape=[layer_4], initializer=tf.zeros_initializer())
      layer_4_output = tf.nn.conv2d(layer_3_output, weights,strides=[1, 1, 1, 1], padding="VALID")
      layer_4_output = tf.nn.bias_add(layer_4_output, biases)
      layer_4_output = tf.nn.relu(layer_4_output)

    print("con4 = {}".format(layer_4_output.shape))

    with tf.variable_scope('layer_5',reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name="weights5", shape=[3, 3,layer_4 , layer_5], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases5", shape=[layer_5], initializer=tf.zeros_initializer())
        layer_5_output = tf.nn.conv2d(layer_4_output, weights,strides=[1, 2, 2, 1], padding="VALID")
        layer_5_output = tf.nn.bias_add(layer_5_output, biases)
        layer_5_output = tf.nn.relu(layer_5_output)

    print("con5 = {}".format(layer_5_output.shape))

    with tf.variable_scope('layer_6',reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name="weights6", shape=[3, 3,layer_5 , layer_6], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases6", shape=[layer_6], initializer=tf.zeros_initializer())
        layer_6_output = tf.nn.conv2d(layer_5_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_6_output = tf.nn.bias_add(layer_6_output, biases)
        layer_6_output = tf.nn.relu(layer_6_output)

    print("con6 = {}".format(layer_6_output.shape))

    with tf.variable_scope('layer_7',reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name="weights7", shape=[3, 3,layer_6 , layer_7], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases7", shape=[layer_7], initializer=tf.zeros_initializer())
        layer_7_output = tf.nn.conv2d(layer_6_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_7_output = tf.nn.bias_add(layer_7_output, biases)
        layer_7_output = tf.nn.relu(layer_7_output)

    print("con7 = {}".format(layer_7_output.shape))

    with tf.variable_scope('layer_8',reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(name="weights8", shape=[3, 3,layer_7 , layer_8], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases8", shape=[layer_8], initializer=tf.zeros_initializer())
        layer_8_output = tf.nn.conv2d(layer_7_output, weights,strides=[1, 1, 1, 1], padding="VALID")
        layer_8_output = tf.nn.bias_add(layer_8_output, biases)
        layer_8_output = tf.nn.relu(layer_8_output)
    print("con8 = {}".format(layer_8_output.shape))


    #dropout layers
    with tf.variable_scope('layer_9',reuse=tf.AUTO_REUSE):
        layer_9_output = tf.layers.dropout(layer_8_output,rate=0.2)
    print("dropout1 = {}".format(layer_9_output.shape))


    #flaten layer

    with tf.variable_scope('layer_10',reuse=tf.AUTO_REUSE):
        layer_10_output=tf.contrib.layers.flatten(layer_9_output)
    print("flaten = {}".format(layer_10_output.shape))

    #dense laye

    with tf.variable_scope('layer_11',reuse=tf.AUTO_REUSE):
        layer_11_output=tf.layers.dense(layer_10_output,512,activation=tf.nn.relu)
    print("dense1 = {}".format(layer_11_output.shape))


    with tf.variable_scope('layer_12',reuse=tf.AUTO_REUSE):
        layer_12_output=tf.layers.dense(layer_11_output,512,activation=tf.nn.relu)
    print("dense2 = {}".format(layer_12_output.shape))


    #dropout layers
    with tf.variable_scope('layer_13',reuse=tf.AUTO_REUSE):
        layer_13_output = tf.layers.dropout(layer_12_output,rate=0.5)
    print("dropout2 = {}".format(layer_13_output.shape))


    with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
        prediction = tf.layers.dense(layer_13_output, 3)
        prediction = tf.nn.softmax(prediction)
    print("output= {}".format(prediction.shape))
    return prediction
