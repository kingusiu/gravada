import tensorflow as tf
import models.layers as lays


class GraphAutoencoder(tf.keras.Model):

    def __init__(self, nodes_n, feat_sz, activation, **kwargs):
        super(GraphAutoencoder, self).__init__(**kwargs)
        self.nodes_n = nodes_n
        self.feat_sz = feat_sz
        self.input_shape_feat = [self.nodes_n, self.feat_sz]
        self.input_shape_adj = [self.nodes_n, self.nodes_n]
        self.activation = activation
        self.encoder = self.build_encoder()
        self.decoder = lays.InnerProductDecoder(z_sz=1, activation=tf.nn.sigmoid)


    def build_encoder(self):
        ''' reduce feat_sz to 1 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat
        #feat_sz-1 layers needed to reduce to R^1 
        for output_sz in reversed(range(1, self.feat_sz)):
            x = lays.GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj)
        return tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=x)


    def call(self, x, adjacency):
        z = self.encoder(x, adjacency)
        adj_reco = self.decoder(z)
        return adj_reco


class GraphVariationalAutoencoder(GraphAutoencoder):

    def build_encoder(self):

        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat

        for output_sz in reversed(range(2, self.feat_sz)):
            x = lays.GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj)

        ''' make latent space params mu and sigma in last compression to feat_sz = 1 '''
        self.z_mean = lays.GraphConvolution(output_sz=1, activation=self.activation)(x, inputs_adj)
        self.z_log_var = lays.GraphConvolution(output_sz=1, activation=self.activation)(x, inputs_adj)

        self.z = self.z_mean + tf.random_normal(self.nodes_n) * tf.exp(self.z_log_std)

        return tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=self.z)
