import tensorflow as tf
import numpy as np
#1a
x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])

out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))



#1b
def f3():
    return tf.constant(0)
x = tf.random_uniform([],minval=-1,maxval=1,dtype=tf.int32)
y = tf.random_uniform([],minval=-1,maxval=1,dtype=tf.int32)

out = tf.case({tf.less(x,y):lambda:tf.add(x,y),tf.greater(x,y):lambda:tf.subtract(x,y)},default=f3,exclusive=True)


#1c
x = tf.constant([[0, -2, -1], [0, 1, 2]] )
y = tf.zeros_like(x)
out = tf.equal(x,y)

print x.get_shape()
print y.get_shape()

init = tf.global_variables_initializer()

#1d
x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
30.97266006,  26.67541885,  38.08450317,  20.74983215,
34.94445419,  34.45999146,  29.06485367,  36.01657104,
27.88236427,  20.56035233,  30.20379066,  29.51215172,
33.71149445,  28.59134293,  36.05556488,  28.66994858])

out = tf.where(tf.less(x,30))
y = tf.gather(x,out)


#1e
x = tf.range(6)+1
out = tf.diag(x)

#1f
array = np.random.randn(10, 10)
x = tf.constant(array)
out = tf.matrix_determinant(x)
out = tf.matrix_determinant(x)

#1g
#Keep in mind that tf.unique() returns a tuple.
x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
y,out =tf.unique(x)#y,idx

#1h
x = tf.truncated_normal([300])
y = tf.truncated_normal([300])

z = tf.subtract(x,y)
avg = tf.reduce_mean(z)
cond = tf.less(avg,0)
out1 = tf.square(z)
out2 = tf.abs(z)

out = tf.where(cond,out1,out2)

sess = tf.Session()
sess.run(init)
print sess.run([y,out])


#print sess.run([y])

