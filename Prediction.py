import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split

# Load the data
X = np.load('X_Test.npy')
Y = np.load('Y_Test.npy')

#initial value of weights

layer_size_1 = 200 # put the same number as it is in neural newtwork code
layer_size_2 = 10

# Load the variables
w_1 = np.load('W1.npy')
w_2 = np.load('W2.npy')
w_3 = np.load('W3.npy')

# Model
def model(X, w_1, w_2,w_3):
    h1 = tf.nn.tanh(tf.matmul(X, w_1))
    h2 = tf.nn.tanh(tf.matmul(h1, w_2))
    return tf.matmul(h2, w_3)

# Run the optimisation
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    Predicted = sess.run(tf.nn.sigmoid(model),feed_dict={X:X, w_1:W1, w_2:W2, w_3:W3})
    Predicted = np.round(Predicted)
    print('Testing', np.mean(Predicted == Y), sum(Y) / len(Y))


