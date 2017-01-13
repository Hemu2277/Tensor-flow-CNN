import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split

# Load the data
X = np.load('X_Train.npy')
Y = np.load('Y_Train.npy')

# Split the data into training and testing
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.75,random_state=1)

# Set up the model
X = tf.placeholder('float', [None, np.shape(x_train)[1]])
Y = tf.placeholder('float', [None, np.shape(y_train)[1]])

#initial value of weights

layer_size_1 = 100
layer_size_2 = 50

w_1 = tf.Variable(tf.random_uniform([np.shape(x_train)[1],layer_size_1],-1,1)) # hidden layer 1
w_2 = tf.Variable(tf.random_uniform([layer_size_1,layer_size_2],0,1)) # hidden layer 2
w_3 = tf.Variable(tf.random_uniform([layer_size_2,np.shape(y_train)[1]],0,1)) # Output layer

# Model
def model(X, w_1, w_2,w_3):
    h1 = tf.nn.tanh(tf.matmul(X, w_1))
    h2 = tf.nn.sigmoid(tf.matmul(h1, w_2))
#    h1 = tf.nn.relu(tf.matmul(X, w_1))
#    h2 = tf.nn.relu(tf.matmul(h1, w_2))
    return tf.matmul(h2, w_3)

# Optimisation step
M = model(X, w_1, w_2,w_3)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(M,Y))
train_op = tf.train.AdadeltaOptimizer(0.05).minimize(cost)
pred = tf.nn.sigmoid(model(X,w_1,w_2,w_3))

batchsize = 2000 # We need not have a batch size as the data size is very less

# Run the optimisation
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(10000):
        p = np.random.permutation(range(len(x_train)))
        trX, trY = x_train[p], y_train[p]
        for j in range(0, len(x_train), batchsize):
            last = j + batchsize
            sess.run(train_op, feed_dict={X: trX[j:j + batchsize], Y: trY[j:j + batchsize]})
        if i%1000 == 0:
            print(sess.run(cost, feed_dict={X: x_train, Y: y_train}))

    Predicted = sess.run(pred,feed_dict={X: x_test, w_1:sess.run(w_1),w_2:sess.run(w_2),w_3:sess.run(w_3)})
    Predicted1 = sess.run(pred, feed_dict={X: x_train, w_1: sess.run(w_1), w_2: sess.run(w_2), w_3: sess.run(w_3)})
    Predicted = np.round(Predicted)
    Predicted1 = np.round(Predicted1)
    print(np.mean(Predicted == y_test), float(sum(y_test)[0]) / float(len(y_test)))
    print(np.mean(Predicted1 == y_train), float(sum(y_train)[0]) / float(len(y_train)))
