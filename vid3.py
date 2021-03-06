import tensorflow as tf

A = tf.Variable([2.0], dtype=tf.float32)
B = tf.Variable([-1.0], dtype=tf.float32)
x = tf.placeholder(tf.float32)

Salida = A * x + B

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print('\n\n')
print(sess.run([A, B]))
print(sess.run(Salida, {x:[1,2,3,4]}))

Ax = tf.assign(A,A + 1) #A = 3
Bx = tf.assign(B,B - 2) #B = -3

print(sess.run([Ax, Bx]))
print(sess.run(Salida, {x:[1,2,3,4]}))
