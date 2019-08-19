import tensorflow as tf

"""Tensor has two parts: """

"""Computational Graph"""
x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)
print(result)


"""The Session"""
"""Automatically closes after this is dont"""
with tf.Session() as sess:
    output = sess.run(result)
    print(sess.run(output))

print(output)
