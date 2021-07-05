import tensorflow as tf
import networkx as nx
import numpy as np

import data.input_samples as inpu
import models.graph_nn as grap



karate_example = True

if karate_example:

    club_dict = dict('Mr. Hi': 0, 'Officer': 1)

    # input sample
    G = nx.karate_club_graph()
    X = np.eye(G.number_of_nodes(), dtype=np.float32) # featureless graph
    A = nx.to_numpy_array(G)
    A_tilde = inpu.normalized_adjacency(A)
    club_labels = nx.get_node_attributes(G,'club')
    y = np.array([club_dict[label] for label in club_labels ])

    # model 
    gnn = grap.GraphAutoencoderKarate(nodes_n=G.number_of_nodes(), feat_sz=1, activation=tf.nn.tanh)
    gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    gnn.fit((X, A_tilde), A, epochs=100, validation_data=((X,A_tilde),A), callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)])

    embedding = gnn((X, A_tilde))

else:

    # get toy data
    X, A, A_tilde = inpu.make_toy_graph()
    pos_weight = float(A.shape[1] * A.shape[2] - A.sum()) / A.sum() # share of positive weight edges

    gnn = grap.GraphAutoencoder(nodes_n=X.shape[-2], feat_sz=X.shape[-1], activation=tf.nn.tanh)
    gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    # gnn.fit(X, A, epochs=100, validation_data=(X,A), callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)])
    gnn.fit((X, A_tilde), A, epochs=100, validation_data=((X,A_tilde),A), callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)])

    A_hat = gnn((X, A_tilde))
