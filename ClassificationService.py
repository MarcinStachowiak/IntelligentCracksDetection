""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
import pandas as pd
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

from PropertiesSupport import PropertiesSupport
from PropertiesKeys import PropertiesKeys
import tensorflow as tf
import os
import numpy as np

prop = PropertiesSupport.load_properties_img_processing()
file_path = os.path.join(os.getcwd(), prop[PropertiesKeys.FILE_WITH_FEATURES])
df=pd.read_csv(file_path,header=None,sep=';')

df_input=df.ix[:,0:df.shape[1]-2]
df_output=df.ix[:,df.shape[1]-1]

df_input_norm=(df_input-df_input.min())/(df_input.max()-df_input.min())

df_output_norm=np.empty((0, 8), int)
for v in df_output.values:
    vector=np.array([0,0,0,0,0,0,0,0])
    vector[int(v)]=1
    df_output_norm = np.vstack([df_output_norm, vector])
df_output_norm=pd.DataFrame(df_output_norm)

msk = np.random.rand(len(df_input_norm)) < 0.75

train_input = df_input_norm[msk]
train_output = df_output_norm[msk]
test_input = df_input_norm[~msk]
test_output = df_output_norm[~msk]

def provide_next_train_data(batch_size):
    random_indexes=np.random.random_integers(0, len(train_input)-1, batch_size)
    train=train_input.ix[random_indexes, :]
    test=train_output.ix[random_indexes, :]
    return train_input.values,train_output.values


#a,b=provide_next_train_data(2)
#mnist = input_data.read_data_sets("C:\\Users\\Marcin\\Downloads\\comp", one_hot=True)
# Parameters
learning_rate = 0.5
num_steps = 700
batch_size = 1280
display_step = 50

# Network Parameters
n_hidden_1 = 556 # 1st layer number of neurons
n_hidden_2 = 156 # 2nd layer number of neurons
num_input = 137 # MNIST data input (img shape: 28*28)
num_classes = 8 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = provide_next_train_data(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_input.values,
                                      Y: test_output.values}))