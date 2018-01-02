import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Get Mnist Data
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Variable
learning_rate = 1e-3
num_units = 256
num_layer = 3
input_size = 28
time_step = 28
total_steps = 2000
category_num = 10
steps_per_validate = 100
steps_per_test = 500
batch_size = tf.placeholder(tf.int32, [])
keep_prob = tf.placeholder(tf.float32, [])


# Get RNN Cell
def cell(num_units):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
    return DropoutWrapper(cell, output_keep_prob=keep_prob)


# Initial
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])
x_shape = tf.reshape(x, [-1, time_step, input_size])

# RNN Layers
cells = tf.nn.rnn_cell.MultiRNNCell([cell(num_units) for _ in range(num_layer)])
h0 = cells.zero_state(batch_size, dtype=tf.float32)
output, hs = tf.nn.dynamic_rnn(cells, inputs=x_shape, initial_state=h0)
output = output[:, -1, :]
# Or h = hs[-1].h

# Output Layer
w = tf.Variable(tf.truncated_normal([num_units, category_num], stddev=0.1), dtype=tf.float32)
b = tf.Variable(tf.constant(0.1, shape=[category_num]), dtype=tf.float32)
y = tf.matmul(output, w) + b

# Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Prediction
correction_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(total_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_label: batch_y, keep_prob: 0.5, batch_size: batch_x.shape[0]})
        # Train Accuracy
        if step % steps_per_validate == 0:
            print('Train', step, sess.run(accuracy, feed_dict={x: batch_x, y_label: batch_y, keep_prob: 0.5,
                                                               batch_size: batch_x.shape[0]}))
        # Test Accuracy
        if step % steps_per_test == 0:
            test_x, test_y = mnist.test.images, mnist.test.labels
            print('Test', step,
                  sess.run(accuracy, feed_dict={x: test_x, y_label: test_y, keep_prob: 1, batch_size: test_x.shape[0]}))
