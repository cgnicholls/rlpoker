import tensorflow as tf


class TBSummariser:

    def __init__(self, scalar_names):
        self.placeholders = {name: tf.placeholder(dtype=tf.float32, name=name) for name in scalar_names}
        self.scalars = {name: tf.summary.scalar(name, placeholder) for name, placeholder in self.placeholders.items()}

        self.merged = tf.summary.merge([v for v in self.scalars.values()])

    def summarise(self, sess, scalar_values):

        feed_dict = {
            self.placeholders[name]: scalar_values[name] for name in self.placeholders
        }
        return sess.run(self.merged, feed_dict=feed_dict)
