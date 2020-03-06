"""
Original file is located at
    https://colab.research.google.com/drive/1ZFAad7MSlT0iS2t8zIC-qieQMvcsb89A
"""

!git clone https://github.com/adhammm/test.git

cd test

import tensorflow as tf
import os
import numpy as np
import csv
import cv2
from ImitationArchitecture import model

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_data():
    # Load training data set from CSV file
	# Pull out columns for X (Image_paths) and Y (Labels)
    with open("data_file.csv", 'r') as read_obj:
        csv_reader = csv.DictReader(read_obj, delimiter=',')

        X_training = []
        Y_training = []
		#Appending a label for each command & a Image_path for each image
        for line in csv_reader:
            X_training.append(line['Images'])

            if ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [0, 1, 1] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [0, 0, 1]):
                Y_training.append(0)
            elif ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 0, 1] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 1, 1]):
                Y_training.append(1)
            elif ([int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 1, 0] or
                    [int(line['Right']), int(line['Forword']), int(line['Left'])] == [1, 0, 0]):
                Y_training.append(2)

    X_training = np.array(X_training)
    Y_training = np.array(Y_training)

    return (X_training, Y_training)

def preprocess_data(imgs_paths, label):
   
    target_size = [288, 352]
	# read the image from file.
    im = tf.io.read_file(imgs_paths)
	# Decode the image to get array of its pixels as a tensor.
    im = tf.image.decode_png(im, channels=3)
	#cast to float32.
    im = tf.cast(im, tf.float32)
	# Resize the image tensor.
    im = tf.image.resize_images(im, target_size)
	# Normalize the image to [0 : +1].
    im = im / 255.0
    return im, label

def data_layer(data_tensor, num_threads=4, prefetch_buffer=50, batch_size=32):
    with tf.variable_scope("data",reuse=tf.AUTO_REUSE):
		# read the data from data tensors
        dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
		# shuffle the data with a buffer and repeat the data 2 times.
        dataset = dataset.shuffle(buffer_size=200).repeat()
		# dataset.map: Applies preprocess_data function to each element of this dataset,
        # and returns a new dataset containing the transformed elements, in the same order as they appeared in the input.
        dataset = dataset.map(preprocess_data, num_parallel_calls=num_threads)
		# dataset.batch: Combines sequential elements of this dataset into batches.
        dataset = dataset.batch(batch_size)
		# dataset.prefetch: will start a background thread to populate a ordered buffer.
        dataset = dataset.prefetch(prefetch_buffer)
		# iterator return us the next batch of data every time we call it.
        iterator = dataset.make_one_shot_iterator()
    return iterator

def loss_functions(logits, labels, num_classes=3):
    with tf.variable_scope("loss",reuse=tf.AUTO_REUSE):
		# Transform our labels to one hot encoded labels
        target_prob = tf.one_hot(labels, num_classes)
		#Evaluate the loss 
        total_loss = tf.losses.softmax_cross_entropy(target_prob, logits)
        #total_loss = tf.losses.get_total_loss()
    return total_loss


def optimizer_func(total_loss, global_step, learning_rate=0.0002):
    with tf.variable_scope("optimizer",reuse=tf.AUTO_REUSE):
		# Setting the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1)
        optimizer = optimizer.minimize(total_loss, global_step=global_step)
    return optimizer


def performance_metric(logits, labels):
    with tf.variable_scope("performance_metric",reuse=tf.AUTO_REUSE):
		# tf.argmax gives you the index of maximum value along the specified axis.
        preds = tf.argmax(logits, axis=1)
		# Cast the labels to interger
        labels = tf.cast(labels, tf.int64)
		# tf.equal() determines if the element in the first tensor equals the one in the second.
        # We get an array of bools (True and False).
        corrects = tf.equal(preds, labels)
		# tf.reduce_mean sums and averages all the values in the tensor
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy

def train(data_tensor):
	# Iteration number
    global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name="iter_number")

    # training graph
    images, labels = data_layer(data_tensor).get_next()
	# Logits is the output tensor of a classification network,
    # whose content is the unnormalized (not scaled between 0 and 1) probabilities.
    logits = model(images)
    loss = loss_functions(logits, labels)
    optimizer = optimizer_func(loss, global_step)
    accuracy = performance_metric(logits, labels)

    # start training
    num_iter = 10000
    log_iter = 1000
    with tf.Session() as sess:
		# initialize all the variables
        sess.run(tf.global_variables_initializer())
        streaming_loss = 0
        streaming_accuracy = 0
		# create tf.train.saver object to save the model later
        saver = tf.train.Saver(max_to_keep=None)
        
        for i in range(1, num_iter + 1):
			# running the training session
            _, loss_batch, acc_batch = sess.run([optimizer, loss, accuracy])
            streaming_loss += loss_batch
            streaming_accuracy += acc_batch
            
			# print the acc and loss every certain number of iterations 
            if i % log_iter == 0:
                print("Iteration: {}, Streaming loss: {:.4f}, Streaming accuracy: {:.6f}"
                        .format(i, streaming_loss/log_iter, streaming_accuracy/log_iter))
                streaming_loss = 0
                streaming_accuracy = 0
			
			# Save model after training done
            if i % num_iter == 0:
                saver.save(sess,"logs/model.ckpt" , global_step=global_step)
                print("model saved")


if __name__ == "__main__":
	
	# Read the data and print it's shape
    data_train = read_data()

    print(data_train[0].shape, data_train[1].shape)
    print(data_train[1])
	
	# start training
    train(data_tensor = data_train)
