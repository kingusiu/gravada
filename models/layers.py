import tensorflow as tf

''' GC layers adapted from Kipf: https://github.com/tkipf/gae/blob/0ebbe9b9a8f496eb12deb9aa6a62e7016b5a5ac3/gae/layers.py '''

class GraphConvolution(tf.keras.layers.Layer):
    
    def __init__(self, input_sz, output_sz, activation, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.input_sz = input_sz
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[input_shape([-1], self.output_sz)])

    def call(self, inputs, adjacency):
        x = tf.matmul(inputs, self.kernel)
        x = tf.sparse_tensor_dense_matmul(adjacency, x)
        return self.activation(x)

    def get_config(self):
        config = super(GraphConvolution, self).get_config()
        config.update({'input_sz': self.input_sz, 'output_sz': self.output_sz, 'activation': self.activation})
        return config

class InnerProductDecoder(tf.keras.layers.Layer):
    