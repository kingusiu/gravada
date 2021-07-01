import tensorflow as tf
import layers as layers

class GraphAutoencoder(tf.keras.Model):

    def __init__(self, n_nodes, feat_sz, **kwargs):
        super(GraphAutoencoder, self).__init__(**kwargs)
        self.n_nodes = n_nodes
        self.feat_sz = feat_sz
        self.encoder = self.build_encoder()
        self.decoder = layers.InnerProductDecoder(z_sz=1, activation=tf.nn.sigmoid)


    def build_encoder(encoder):
        ''' reduce feat_sz to 1 '''
        inputs = tf.keras.layers.Input(shape=[self.n_nodes, self.feat_sz], dtype=tf.float32, name='encoder_input')
        for i in range(feat_sz-1):
            x = layers.GraphConvolution(input_sz, output_sz, activation)
        return tf.keras.Model(inputs=inputs, outputs=x)


    def call(self, x, adjacency):
        z = self.encoder(x, adjacency)
        adj_reco = self.decoder(z)
        return adj_reco