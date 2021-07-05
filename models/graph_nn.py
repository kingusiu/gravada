import tensorflow as tf
import models.layers as lays


@tf.function
def adjacency_loss_from_logits(adj_orig, adj_pred, pos_weight):
    # cast probability to a_ij = 1 if > 0.5 or a_ij = 0 if <= 0.5
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=adj_pred, labels=adj_orig, pos_weight=pos_weight)) 
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adj_pred, labels=adj_orig)) 


@tf.function
def adjacency_loss(adj_orig, adj_pred):
    # cast probability to a_ij = 1 if > 0.5 or a_ij = 0 if <= 0.5
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(adj_orig, adj_pred) 


class GraphAutoencoder(tf.keras.Model):

    def __init__(self, nodes_n, feat_sz, activation, **kwargs):
        super(GraphAutoencoder, self).__init__(**kwargs)
        self.nodes_n = nodes_n
        self.feat_sz = feat_sz
        self.input_shape_feat = [self.nodes_n, self.feat_sz]
        self.input_shape_adj = [self.nodes_n, self.nodes_n]
        self.activation = activation
        self.encoder = self.build_encoder()
        self.decoder = lays.InnerProductDecoder(activation=tf.keras.activations.linear) # if activation sigmoid -> return probabilities from logits


    def build_encoder(self):
        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat
        #feat_sz-1 layers needed to reduce to R^1 
        for output_sz in reversed(range(2, self.feat_sz)):
            x = lays.GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj)
        # NO activation before latent space: last graph with linear pass through activation
        x = lays.GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
        encoder = tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=x)
        encoder.summary()
        return encoder


    def call(self, inputs):
        z = self.encoder(inputs)
        adj_reco = self.decoder(z)
        return adj_reco


    def train_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.math.reduce_sum(adj_orig)

        with tf.GradientTape() as tape:
            adj_pred = self((X, adj_tilde))  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = adjacency_loss(adj_orig, adj_pred) # TODO: add regularization
            loss = adjacency_loss_from_logits(adj_orig, adj_pred, pos_weight) # TODO: add regularization

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.math.reduce_sum(adj_orig)

        adj_pred = self((X, adj_tilde), training=False)
        # loss = adjacency_loss(adj_orig, adj_pred) # TODO: add regularization
        loss = adjacency_loss_from_logits(adj_orig, adj_pred, pos_weight) # TODO: add regularization
        return {'loss' : loss}



class GraphVariationalAutoencoder(GraphAutoencoder):

    def build_encoder(self):

        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat

        for output_sz in reversed(range(2, self.feat_sz)):
            x = lays.GraphConvolution(output_sz=output_sz, activation=self.activation)(x, inputs_adj)

        ''' make latent space params mu and sigma in last compression to feat_sz = 1 '''
        self.z_mean = lays.GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
        self.z_log_var = lays.GraphConvolution(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)

        self.z = self.z_mean + tf.random_normal(self.nodes_n) * tf.exp(self.z_log_std)

        return tf.keras.Model(inputs=[inputs_feat, inputs_adj], outputs=self.z)


class GraphAutoencoderKarate(GraphAutoencoder):
    ''' test graph autoencoder for karate club example 
        with fixed setup (3 layers, tanh activation)
    '''

     def __init__(self, nodes_n, feat_sz, activation, **kwargs):
        super(GraphAutoencoderKarate, self).__init__(**kwargs)
        self.activation = tf.nn.tanh


     def build_encoder(self):
        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat
        #feat_sz-1 layers needed to reduce to R^1 
        x = lays.GraphConvolution(output_sz=4, activation=self.activation)(x, inputs_adj)
        x = lays.GraphConvolution(output_sz=4, activation=self.activation)(x, inputs_adj)
        x = lays.GraphConvolution(output_sz=2, activation=self.activation)(x, inputs_adj)
        encoder = tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=x)
        encoder.summary()
        return encoder

