import tensorflow as tf
import models.layers as lays


@tf.function
def adjacency_loss_from_logits(adj_orig, adj_pred, pos_weight):
    # cast probability to a_ij = 1 if > 0.5 or a_ij = 0 if <= 0.5
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=adj_pred, labels=adj_orig, pos_weight=pos_weight)) 
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adj_pred, labels=adj_orig)) 


### Latent Space Loss (KL-Divergence)
@tf.function
def kl_loss(z_mean, z_log_var):
    kl = 1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    return -0.5 * tf.reduce_mean(kl, axis=-1) # multiplying mse by N -> using sum (instead of mean) in kl loss


class GraphAutoencoder(tf.keras.Model):

    def __init__(self, nodes_n, feat_sz, activation=tf.nn.tanh, graphLayer='conv', **kwargs):
        super(GraphAutoencoder, self).__init__(**kwargs)
        self.nodes_n = nodes_n
        self.feat_sz = feat_sz
        self.input_shape_feat = [self.nodes_n, self.feat_sz]
        self.input_shape_adj = [self.nodes_n, self.nodes_n]
        self.activation = activation
        self.loss_fn = tf.nn.weighted_cross_entropy_with_logits
        self.graphLayer = lays.GraphConvolution if graphLayer == 'conv' else lays.GraphConvolutionExpanded
        self.encoder = self.build_encoder()
        self.decoder = lays.InnerProductDecoder(activation=tf.keras.activations.linear) # if activation sigmoid -> return probabilities from logits


    def build_encoder(self):
        ''' reduce feat_sz to 2 '''
        inputs_feat = tf.keras.layers.Input(shape=self.input_shape_feat, dtype=tf.float32, name='encoder_input_features')
        inputs_adj = tf.keras.layers.Input(shape=self.input_shape_adj, dtype=tf.float32, name='encoder_input_adjacency')
        x = inputs_feat
        #feat_sz-1 layers needed to reduce to R^2 
        for output_sz in reversed(range(2, self.feat_sz)):
            x = self.graphLayer(output_sz=output_sz, activation=self.activation)(x, inputs_adj)
        # NO activation before latent space: last graph with linear pass through activation
        x = self.graphLayer(output_sz=1, activation=tf.keras.activations.linear)(x, inputs_adj)
        encoder = tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=x)
        encoder.summary()
        return encoder


    def call(self, inputs):
        z = self.encoder(inputs)
        adj_pred = self.decoder(z)
        return z, adj_pred

    def train_step(self, data):
        # import ipdb; ipdb.set_trace()
        (X, adj_tilde), adj_orig = data
        # pos_weight = zero-adj / one-adj -> no-edge vs edge ratio (if more zeros than ones: > 1, if more ones than zeros < 1, e.g. for 1% of ones: 100)
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        with tf.GradientTape() as tape:
            z, adj_pred = self((X, adj_tilde))  # Forward pass
            # Compute the loss value (binary cross entropy for a_ij in {0,1})
            loss = self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight) # TODO: add regularization

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        z, adj_pred = self((X, adj_tilde), training=False)  # Forward pass
        loss = tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight)) # TODO: add regularization
        
        return {'loss' : loss}




class GraphVariationalAutoencoder(GraphAutoencoder):
    
    def __init__(self, nodes_n, feat_sz, activation, **kwargs):
        super(GraphVariationalAutoencoder, self).__init__(nodes_n, feat_sz, activation, **kwargs)
        self.loss_fn_latent = kl_loss

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

        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(self.z_mean)[0], self.nodes_n, 1))
        self.z = self.z_mean +  epsilon * tf.exp(0.5 * self.z_log_var)

        return tf.keras.Model(inputs=(inputs_feat, inputs_adj), outputs=[self.z, self.z_mean, self.z_log_var])
    
    
    def call(self, inputs):
        # import ipdb; ipdb.set_trace()
        z, z_mean, z_log_var = self.encoder(inputs)
        adj_pred = self.decoder(z)
        return z, z_mean, z_log_var, adj_pred
    
    def train_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)


        with tf.GradientTape() as tape:
            z, z_mean, z_log_var, adj_pred = self((X, adj_tilde))  # Forward pass
            # Compute the loss value (binary cross entropy for a_ij in {0,1})
            loss_reco = tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight), axis=(1,2)) # TODO: add regularization
            loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var), axis=1)
            loss = loss_reco + loss_latent

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}


    def test_step(self, data):
        (X, adj_tilde), adj_orig = data
        pos_weight = tf.cast(adj_orig.shape[1] * adj_orig.shape[2] - tf.math.reduce_sum(adj_orig), tf.float32) / tf.cast(tf.math.reduce_sum(adj_orig), tf.float32)

        z, z_mean, z_log_var, adj_pred = self((X, adj_tilde))  # Forward pass
        # Compute the loss value (binary cross entropy for a_ij in {0,1})
        loss_reco =  tf.math.reduce_mean(self.loss_fn(labels=adj_orig, logits=adj_pred, pos_weight=pos_weight)) # TODO: add regularization
        loss_latent = tf.math.reduce_mean(self.loss_fn_latent(z_mean, z_log_var))
        return {'loss' : loss_reco+loss_latent, 'loss_reco': loss_reco, 'loss_latent': loss_latent}



@tf.function
def masked_karate_loss(y_true, y_pred, mask):
    scce = tf.keras.losses.SparseCategoricalCrossentropy() # using sparseCat since providing labels as integers
    return scce(y_true[mask], y_pred[mask])


class GraphClassifierKarate(GraphAutoencoder):
    ''' test graph classification encoder for karate club example 
        with fixed setup (3 layers, tanh activation)
    '''

    def __init__(self, nodes_n, feat_sz, activation=tf.nn.tanh, **kwargs):
        super(GraphClassifierKarate, self).__init__(nodes_n, feat_sz, activation, **kwargs)


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

    def call(self, inputs):
        z = self.encoder(inputs)
        probs = tf.nn.softmax(z)
        return z, probs

    def train_step(self, data):
        # import ipdb; ipdb.set_trace()
        (X, adj_tilde, mask), Y = data

        with tf.GradientTape() as tape:
            z, probs = self((X, adj_tilde))  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = adjacency_loss(adj_orig, adj_pred) # TODO: add regularization
            loss = masked_karate_loss(Y, probs, mask) # TODO: add regularization

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        (X, adj_tilde, mask), Y = data

        z, probs = self((X, adj_tilde), training=False)
        loss = masked_karate_loss(Y, probs, mask) # TODO: add regularization
        return {'loss' : loss}


@tf.function
def masked_karate_adjacency_loss(adj_orig, adj_pred, mask):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return bce(adj_orig[mask], adj_pred[mask])    


class GraphAutoencoderKarate(GraphClassifierKarate):

    def call(self, inputs):
        # import ipdb; ipdb.set_trace()
        z = self.encoder(inputs)
        adj_pred = self.decoder(z)
        return z, adj_pred

    def train_step(self, data):
        # import ipdb; ipdb.set_trace()
        (X, adj_tilde, mask), adj_orig = data

        with tf.GradientTape() as tape:
            z, adj_pred = self((X, adj_tilde))  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = adjacency_loss(adj_orig, adj_pred) # TODO: add regularization
            loss = masked_karate_adjacency_loss(adj_orig, adj_pred, mask) # TODO: add regularization

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        (X, adj_tilde, mask), adj_orig = data

        z, adj_pred = self((X, adj_tilde), training=False)  # Forward pass
        loss = masked_karate_adjacency_loss(adj_orig, adj_pred, mask) # TODO: add regularization
        
        return {'loss' : loss}
