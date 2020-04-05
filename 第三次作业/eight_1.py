import tensorflow as tf
x=[ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]

y=[ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]

x_z=tf.constant(x,tf.float32)
y_z=tf.constant(y,tf.float32)
x_j=tf.reduce_mean(x_z)
y_j=tf.reduce_mean(y_z)

xz=x_z-x_j
yz=y_z-y_j
zm=xz*yz
zs=xz*xz

w=tf.reduce_sum(zm)/tf.reduce_sum(zs)
b=y_j-w*x_j
sess = tf.Session()  
print("w="+str(sess.run(w)))
print("b="+str(sess.run(b)))