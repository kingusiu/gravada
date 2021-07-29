import tensorflow as tf

''' GC layers adapted from Kipf: https://github.com/tkipf/gae/blob/0ebbe9b9a8f496eb12deb9aa6a62e7016b5a5ac3/gae/layers.py '''

class GraphConvolution(tf.keras.layers.Layer):
    
    def __init__(self, output_sz, activation=tf.keras.activations.linear, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_sz = output_sz
        self.activation = activation

    def build(self, input_shape):
        # build is invoked first time the layer is called, input_shape is based on the first argument 
        # passed to call that is stripped from args & kwargs as 'inputs': https://github.com/tensorflow/tensorflow/blob/b36436b087bd8e8701ef51718179037cccdfc26e/tensorflow/python/keras/engine/base_layer.py#L981-L982
        
        # kernel in Keras is transposed: instead of Wx computing x^T W^T, s.t. first dimension of W matches input dimension
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        self.bias = self.add_weight("bias", shape=[self.output_sz], initializer=tf.keras.initializers.Zeros())


    def call(self, inputs, adjacency):
        # WX
        x = tf.matmul(inputs, self.kernel)
        x = tf.add(x, self.bias)
        # AWX
        x = tf.matmul(adjacency, x)
        #sig(AWX)
        return self.activation(x)

    def get_config(self):
        config = super(GraphConvolution, self).get_config()
        config.update({'output_sz': self.output_sz, 'activation': self.activation})
        return config


class GraphConvolutionExpanded(GraphConvolution):
    ''' graph convolution with diagonalized features NxN and 1x1 convolution of channels 
        adapted from https://arxiv.org/pdf/1704.01212.pdf
    '''

    def __init__(self, use_global_bias=False, **kwargs):
        super(GraphConvolutionExpanded, self).__init__(**kwargs)
        self.use_global_bias = use_global_bias


    def build(self, input_shape):

        feats_n = input_shape[-1]
        nodes_n = input_shape[-2]
        
        # elementwise (NxF) multiplication kernel
        self.feat_kernel = self.add_weight("feat_kernel", shape=[nodes_n, feats_n], initializer=tf.keras.initializers.GlorotUniform())
        self.feat_bias = self.add_weight("feat_bias", shape=[nodes_n, feats_n], initializer=tf.keras.initializers.Zeros())
        # K wise reduction kernel (k output features)
        self.reduce_kernel = self.add_weight("redu_kernel", shape=[feats_n, self.output_sz], initializer=tf.keras.initializers.GlorotUniform())
        self.reduce_bias = self.add_weight("redu_bias", shape=[self.output_sz], initializer=tf.keras.initializers.Zeros())

        # global bias added after message passing
        if self.use_global_bias:
            self.global_bias = self.add_weight("glob_bias", shape=[self.output_sz], initializer=tf.keras.initializers.Zeros())
        else:
            self.global_bias = None


    def call(self, inputs, adjacency):
        # diag(X)*diag(W)
        x = inputs * self.feat_kernel + self.feat_bias
        # reduce 
        x = tf.matmul(x, self.reduce_kernel)
        x = tf.add(x, self.reduce_bias)
        # AWX
        x = tf.matmul(adjacency, x)
        #sig(AWX)
        return self.activation(x)



class InnerProductDecoder(tf.keras.layers.Layer):

    ''' inner product decoder reconstructing adjacency matrix as act(z^T z) 
        input assumed of shape [batch_sz x n_nodes x z_d]
        where 
            batch_sz can be 1 for single example feeding
            n_nodes ... number of nodes in graph
            z_d ... dimensionality of latent space
    '''

    def __init__(self, activation=tf.keras.activations.linear, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.activation = activation

    def call(self, inputs):
        perm = [0, 2, 1] if len(inputs.shape) == 3 else [1, 0]
        z_t = tf.transpose(inputs, perm=perm)
        adjacency_hat = tf.matmul(inputs, z_t)
        return self.activation(adjacency_hat)


    def get_config(self):
        config = super(InnerProductDecoder, self).get_config()
        return config
