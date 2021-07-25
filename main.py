import tensorflow as tf
import networkx as nx
import numpy as np
import h5py

import data.input_samples as inpu
import models.graph_nn as grap



karate_classify_example = False
karate_autoencode_example = False

if karate_classify_example:

    # data
    X, A_tilde, y, train_mask, valid_mask = inpu.make_karate_data_classifier()
    nodes_n = len(y)

    # model 
    gnn = grap.GraphClassifierKarate(nodes_n=nodes_n, feat_sz=nodes_n, activation=tf.nn.tanh)
    gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), run_eagerly=True)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)]
    # have to add dummy batch dimension, otherwise keras fit will cut off data
    gnn.fit((X[np.newaxis,:,:], A_tilde[np.newaxis,:,:], train_mask[np.newaxis,:]), y[np.newaxis,:], epochs=100, validation_data=((X[np.newaxis,:,:], A_tilde[np.newaxis,:,:], valid_mask[np.newaxis,:]), y[np.newaxis,:]), callbacks=callbacks)

    z, probs = gnn((X, A_tilde))


elif karate_autoencode_example:

    # data
    X, A_tilde, A, train_mask, valid_mask = inpu.make_karate_data_autoencoder()
    nodes_n = len(X)

    # model 
    gnn = grap.GraphAutoencoderKarate(nodes_n=nodes_n, feat_sz=nodes_n, activation=tf.nn.tanh)
    gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), run_eagerly=True)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)]
    # have to add dummy batch dimension, otherwise keras fit will cut off data
    # here the target is the original adjacency matrix
    gnn.fit((X[np.newaxis,:,:], A_tilde[np.newaxis,:,:], train_mask[np.newaxis,:,:]), A[np.newaxis,:,:], epochs=100, validation_data=((X[np.newaxis,:,:], A_tilde[np.newaxis,:,:], valid_mask[np.newaxis,:,:]), A[np.newaxis,:,:]), callbacks=callbacks)

    z, adj_pred = gnn((X, A_tilde))




else:

    # load data
    filename = '/home/kinga/dev/datasamples/L1_anomaly_challenge/background_training_500K.h5'
    ff = h5py.File(filename, 'r')
    particles = np.asarray(ff.get('Particles'))

    nodes_n = particles.shape[1]
    feat_sz = particles.shape[2]
    batch_size = 128

    particles_train = particles[:batch_size*20] # have to take multiple of batch_size because decoder output has trailing dim 1 and callbacks can not handle (alternative: squeeze latent outputs)

    A = inpu.make_adjacencies(particles_train)
    A_tilde = inpu.normalized_adjacency(A)

    particles_train = inpu.normalize_features(particles_train)

    gnn = grap.GraphVariationalAutoencoder(nodes_n=nodes_n, feat_sz=feat_sz, activation=tf.nn.tanh)
    # gnn = grap.GraphAutoencoder(nodes_n=nodes_n, feat_sz=feat_sz, activation=tf.nn.tanh)
    gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), run_eagerly=True)

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)]
    gnn.fit((particles_train, A_tilde), A, epochs=100, batch_size=batch_size, validation_split=0.25, callbacks=callbacks)
