#	Tensor:

#	6. # Tensor de rango 0
#	[5., 12., 1., 0.] # Tensor de rango 1 > [4]
#	[[1., 2., 1.], [3., 3., 2.]] # Tensor de rango 2 > [2, 3]
#	[[[3., 3., 1.]], [[7., 5., 49.]]] # Tensor de rango 3  > [2, 1, 3]

import tensorflow as tf

# nodo 1
nodo1 = tf.constant(3.0, dtype=tf.float32)

# node 2
nodo2 = tf.constant(6.0)

print(nodo1)
print(nodo2)

sess = tf.Session()
print(sess.run([nodo1,nodo2]))
