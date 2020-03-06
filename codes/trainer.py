'''
Original file is located at
    https://colab.research.google.com/drive/1x7T67DJ0M7oP-0on20p8xy9xPa-1_hMI


!git clone https://github.com/adhammm/test.git

cd test
'''
import tensorflow as tf
import numpy as np
import csv
import cv2
from network import model

def read_data():
    # Load training data set from CSV file
    with open("data_file.csv", 'r') as read_obj:
        csv_reader = csv.DictReader(read_obj, delimiter=',')

        X_training = []
        Y_training = []
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
    # read the image from file
    target_size = [288, 352]
    im = tf.io.read_file(imgs_paths)
    im = tf.image.decode_png(im, channels=3)
    im = tf.cast(im, tf.float32)
    im = tf.image.resize_images(im, target_size)
    im = im / 255.0
    return im, label

def data_layer(data_tensor, num_threads=8, prefetch_buffer=100, batch_size=128):
    with tf.variable_scope("data",reuse=tf.AUTO_REUSE):
        dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
        dataset = dataset.shuffle(buffer_size=800).repeat()
        dataset = dataset.map(preprocess_data, num_parallel_calls=num_threads)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_buffer)
        iterator = dataset.make_one_shot_iterator()
    return iterator

def loss_functions(logits, labels, num_classes=3):
    with tf.variable_scope("loss",reuse=tf.AUTO_REUSE):
        target_prob = tf.one_hot(labels, num_classes)
        total_loss = tf.losses.softmax_cross_entropy(target_prob, logits)
        total_loss = tf.losses.get_total_loss()
    return total_loss


def optimizer_func(total_loss, global_step, learning_rate=0.0002):
    with tf.variable_scope("optimizer",reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1)
        optimizer = optimizer.minimize(total_loss, global_step=global_step)
    return optimizer


def performance_metric(logits, labels):
    with tf.variable_scope("performance_metric",reuse=tf.AUTO_REUSE):
        preds = tf.argmax(logits, axis=1)
        labels = tf.cast(labels, tf.int64)
        corrects = tf.equal(preds, labels)
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy

def train(data_tensor):
    global_step = tf.Variable(1, dtype=tf.int32, trainable=False, name="iter_number")

    # training graph
    images, labels = data_layer(data_tensor).get_next()
    logits = model(images)
    loss = loss_functions(logits, labels)
    optimizer = optimizer_func(loss, global_step)
    accuracy = performance_metric(logits, labels)

    # start training
    num_iter = 1000
    log_iter = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        streaming_loss = 0
        streaming_accuracy = 0
        saver = tf.train.Saver(max_to_keep=None)
        print(images)

        for i in range(1, num_iter + 1):
            _, loss_batch, acc_batch = sess.run([optimizer, loss, accuracy])
            streaming_loss += loss_batch
            streaming_accuracy += acc_batch
            if i % log_iter == 0:
                print("Iteration: {}, Streaming loss: {:.4f}, Streaming accuracy: {:.6f}"
                        .format(i, streaming_loss/log_iter, streaming_accuracy/log_iter))
                streaming_loss = 0
                streaming_accuracy = 0

            if i % num_iter == 0:
                saver.save(sess,"logs/model.ckpt" , global_step=global_step)
                print("model saved")


if __name__ == "__main__"":

    data_train = read_data()

    print(data_train[0].shape, data_train[1].shape)
    print(data_train[1])

    train(data_tensor = data_train)
