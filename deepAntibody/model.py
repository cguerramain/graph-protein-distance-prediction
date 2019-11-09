import tensorflow as tf
from ops_ import e2e, lrelu, batch_norm


class ProteinGCNN:
    def __init__(self, sess, batch_size=32, matrix_shape=(64, 64), num_bins=32):
        self.sess = sess
        self.batch_size = batch_size
        self.matrix_shape = matrix_shape
        self.num_bins = num_bins

        self.build_model()

    def build_model(self):
        self.input_ = tf.placeholder(tf.uint16,
                                     [self.batch_size, self.matrix_shape[0],
                                      self.matrix_shape[1], self.num_bins],
                                     name='real_distance_matrix')
        h0 = lrelu(e2e())

