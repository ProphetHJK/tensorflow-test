import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    a = tf.get_variable("a", [2], initializer=tf.ones_initializer())
    b = tf.get_variable("b", [2], initializer=tf.zeros_initializer())
    print(a)
    print(b)
g2 = tf.Graph()
with g2.as_default():
    b = tf.get_variable("b", [2], initializer=tf.ones_initializer())
    a = tf.get_variable("a", [2], initializer=tf.zeros_initializer())

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()

    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("a")))
        print(sess.run(tf.get_variable("b")))


with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()

    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("a")))
        print(sess.run(tf.get_variable("b")))