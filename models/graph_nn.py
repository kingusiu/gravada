import tensorflow as tf
import layers as layers


class GraphAutoencoder(tf.keras.Model):

    def __init__(self, n_nodes, feat_sz, **kwargs):
        super(GraphAutoencoder, self).__init__(**kwargs)
        self.n_nodes = n_nodes
        self.feat_sz = feat_sz
        self.input_shape = [self.n_nodes, self.feat_sz]
        self.encoder = self.build_encoder()
        self.decoder = layers.InnerProductDecoder(z_sz=1, activation=tf.nn.sigmoid)


    def build_encoder(self):
        ''' reduce feat_sz to 1 '''
        inputs = tf.keras.layers.Input(shape=self.input_shape, dtype=tf.float32, name='encoder_input')
        x = inputs
        #feat_sz-1 layers needed to reduce to R^1 
        for i in range(feat_sz-1):
            x = layers.GraphConvolution(input_sz, output_sz, activation=self.activation)(x)
        return tf.keras.Model(inputs=inputs, outputs=x)


    def call(self, x, adjacency):
        z = self.encoder(x, adjacency)
        adj_reco = self.decoder(z)
        return adj_reco


class GraphVariationalAutoencoder(GraphAutoencoder):

    def build_encoder(self):

        ''' reduce feat_sz to 1 '''
        inputs = tf.keras.layers.Input(shape=[self.n_nodes, self.feat_sz], dtype=tf.float32, name='encoder_input')
        for i in range(feat_sz-2):
            x = layers.GraphConvolution(input_sz, output_sz, activation=self.activation)(x)

        self.z_mean = layers.GraphConvolution(input_sz, output_sz, activation=self.activation)(x)
        self.z_log_var = layers.GraphConvolution(input_sz, output_sz, activation=self.activation)(x)

        self.z = self.z_mean + tf.random_normal(self.n_nodes) * tf.exp(self.z_log_std)

        return tf.keras.Model(inputs=inputs, outputs=self.z)
