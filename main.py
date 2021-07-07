import tensorflow as tf
import networkx as nx
import numpy as np

import data.input_samples as inpu
import models.graph_nn as grap



karate_example = True

if karate_example:

    # data
    X, A_tilde, y, train_mask, valid_mask = inpu.make_karate_data()
    nodes_n = len(y)

    # model 
    gnn = grap.GraphAutoencoderKarate(nodes_n=nodes_n, feat_sz=nodes_n, activation=tf.nn.tanh)
    gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), run_eagerly=True)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)]
    # have to add dummy batch dimension, otherwise keras fit will cut off data
    gnn.fit((X[np.newaxis,:,:], A_tilde[np.newaxis,:,:], train_mask[np.newaxis,:]), y[np.newaxis,:], epochs=100, validation_data=((X[np.newaxis,:,:], A_tilde[np.newaxis,:,:], valid_mask[np.newaxis,:]), y[np.newaxis,:]), callbacks=callbacks)

    z, probs = gnn((X, A_tilde))

else:

    # get toy data
    X, A, A_tilde = inpu.make_toy_graph()
    pos_weight = float(A.shape[1] * A.shape[2] - A.sum()) / A.sum() # share of positive weight edges

    gnn = grap.GraphAutoencoder(nodes_n=X.shape[-2], feat_sz=X.shape[-1], activation=tf.nn.tanh)
    gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    # gnn.fit(X, A, epochs=100, validation_data=(X,A), callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)])
    gnn.fit((X, A_tilde), A, epochs=100, validation_data=((X,A_tilde),A), callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)])

    A_hat = gnn((X, A_tilde))
