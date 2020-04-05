import tensorflow as tf

x=[ 64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]

y=[ 62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]

x=tf.constant(x,tf.float32)
y=tf.constant(y,tf.float32)

n=tf.size(x)
n=tf.to_float(n)
# xy=tf.multiply(x,y)
xy=x*y
# xx=tf.multiply(x,x)
xx=x*x
x_sum=tf.reduce_sum(x)
y_sum=tf.reduce_sum(y)
xy_sum=tf.reduce_sum(xy)
xx_sum=tf.reduce_sum(xx)

w=(n*xx_sum-x_sum*y_sum)/(n*xx_sum-x_sum*x_sum)

b=(y_sum-w*x_sum)/n

sess=tf.Session()
print("w="+str(sess.run(w)))
print("b="+str(sess.run(b)))
