import tensorflow as tf

''' GC layers adapted from Kipf: https://github.com/tkipf/gae/blob/0ebbe9b9a8f496eb12deb9aa6a62e7016b5a5ac3/gae/layers.py '''

class GraphConvolution(tf.keras.layers.Layer):
    
    def __init__(self, output_sz, activation, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        # kernel in Keras is transposed: instead of Wx computing x^T W^T, s.t. first dimension of W matches input dimension
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.output_sz])

    def call(self, inputs, adjacency):
        x = tf.matmul(inputs, self.kernel)
        x = tf.matmul(adjacency, x)
        return self.activation(x)

    def get_config(self):
        config = super(GraphConvolution, self).get_config()
        config.update({'output_sz': self.output_sz, 'activation': self.activation})
        return config


class InnerProductDecoder(tf.keras.layers.Layer):

    ''' inner product decoder reconstructing adjacency matrix as sigma(z^T z) '''

    def __init__(self, z_sz, activation, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.z_sz = z_sz
        self.activation = activation

    def call(self, inputs):
        z_t = tf.transpose(inputs)
        adjacency_hat = tf.matmul(inputs, z_t)
        x = tf.reshape(adjacency_hat, [-1]) # flatten for activation
        x = self.activation(x)
        return tf.reshape(x, adjacency_hat.shape)

    def get_config(self):
        config = super(InnerProductDecoder, self).get_config()
        config.update({'z_sz': self.z_sz})
        return config
