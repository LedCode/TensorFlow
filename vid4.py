import tensorflow as tf
import matplotlib.pyplot as plt

# Variables
W = tf.Variable([.01], dtype=tf.float32)
b = tf.Variable([.01], dtype=tf.float32)
# Entradas y Salidas
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
reg_lineal = W*x + b

# Costo
cost = tf.reduce_sum(tf.square(reg_lineal-y)) 
# Algoritmo del Gradiente Descendente
# 	W:= W - learning_rate*dcost/dW
# 	b:= b - learning_rate*dcost/db
op = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = op.minimize(cost)

# Datos
X_data = [1,2,3,4,5,6,7]
Y_data = [-4.2,-0.9,2.1,5.5,7.7,11.9,12]
# Y = 3*X -7
# Inicializar las variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Entrenamiento
for i in range(1000):
	_, cost_ = sess.run([train, cost], feed_dict={x: X_data, y: Y_data})
	#print(cost_)

W_, b_, cost_ = sess.run([W,b,cost], {x: X_data, y: Y_data})
print("\nValores Actuales:")
print("\tW: %s b: %s cost: %s"%(W_, b_, cost_))

plt.plot(X_data,Y_data,'ro',label='Datos')
plt.plot(X_data,[W_*Xi + b_ for Xi in X_data],label='Ajuste')
plt.legend()
plt.show()
