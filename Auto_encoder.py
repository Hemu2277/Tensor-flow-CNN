import tensorflow as tf
import numpy as np

# Training data
x_train = np.load('X_Data.npy')
# Testing data

# Set up the model
X = tf.placeholder('float', [None, np.shape(x_train)[1]])
Y = tf.placeholder('float', [None, np.shape(x_train)[1]])

#initial value of weights

layer_size_1 = 800
layer_size_2 = 400

w_1 = tf.Variable(tf.random_uniform([np.shape(x_train)[1],layer_size_1],-1,1)) # hidden layer 1
w_2 = tf.Variable(tf.random_uniform([layer_size_1,layer_size_2],0,1)) # hidden layer 2
w_3 = tf.Variable(tf.random_uniform([layer_size_2,layer_size_1],0,1)) # hidden layer 3
w_4 = tf.Variable(tf.random_uniform([layer_size_1,np.shape(x_train)[1]],-1,1)) # Output layer

# Model
def model(X, w_1, w_2,w_3,w_4):
    h1 = tf.nn.tanh(tf.matmul(X, w_1))
    h2 = tf.nn.sigmoid(tf.matmul(h1, w_2))
    h3 = tf.nn.tanh(tf.matmul(h2, w_3))
    return tf.matmul(h3, w_4)

def pred(X,w_1,w_2):
    H1 = tf.nn.sigmoid(tf.matmul(X, w_1))
    return tf.matmul(H1, w_2)

# Optimisation step
M = model(X, w_1, w_2,w_3,w_4)
cost = tf.reduce_mean(tf.square(M-Y))
train_op = tf.train.AdadeltaOptimizer(0.05).minimize(cost)
compressed = tf.nn.sigmoid(pred(X,w_1,w_2))

batchsize = 300 # We need not have a batch size as the data size is very less

# Run the optimisation
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(20000):
        p = np.random.permutation(range(len(x_train)))
        trX, trY = x_train[p], x_train[p]
        for j in range(0, len(x_train), batchsize):
            last = j + batchsize
            sess.run(train_op, feed_dict={X: trX[j:j + batchsize], Y: trY[j:j + batchsize]})
        if i%1000 == 0:
            print(sess.run(cost, feed_dict={X: x_train, Y: x_train}))

    new_features = sess.run(compressed,feed_dict={X: x_train, w_1:sess.run(w_1),w_2:sess.run(w_2)})
    np.save('X_Data_encoded',new_features)