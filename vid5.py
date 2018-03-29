
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,W) + b) 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for loop in range(1000):
	batch_x, batch_y = mnist.train.next_batch(50)
	cost, _ = sess.run([cross_entropy,train_step], feed_dict={x: batch_x, y: batch_y})
	print('loop: ',loop, ' cost: ', cost)	

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('\nExactitud: ', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
