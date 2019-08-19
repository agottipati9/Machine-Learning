import tensorflow as tf
import os
import numpy as np
from TensorDemo import create_feature_sets_and_labels
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

# 10 classes, 0-9
'''
This is what one_hot does:
0 =[1,0,0,0,0,0,0,0,0]
1 =[0,1,0,0,0,0,0,0,0]
'''

'''
input > weight > hidden layer 1 (activation function) > weights > hidden l 2
(activation function) > weights > output layer

compare out to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > mnimize cost (AdamOptimizer... SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch
'''

n_nodes_hl = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

tf.logging.set_verbosity(old_v)

n_classes = 2
"""Only 100 go through at a time"""
batch_size = 100

# height x width the None, 784 acts as filter of data if not of this type tensor throws exception
x = tf.placeholder('float',[None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    ouput_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # input_data * weight + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, ouput_layer['weights']) + ouput_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    #learning_rate default = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #cycles feed forward + backpro
    hm_epochs = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        """Trains the network"""
        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        """After optimized weights use test data"""
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)