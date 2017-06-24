import tensorflow as tf

m1 = tf.constant([[2,2]])
m2 = tf.constant([[3],[3]])

#defining operations
dot_operation = tf.matmul(m1,m2)
#One method to use session
sess = tf.Session()
result = sess.run(dot_operation)
print result
sess.close()
#another method to use session
with tf.Session() as sess:
	result2 = sess.run(dot_operation)
	print result2

