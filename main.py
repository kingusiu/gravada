import tensorflow as tf

import data.input_samples as inpu
import models.graph_nn as grap



# get toy data
X, A, A_tilde = inpu.make_toy_graph()
pos_weight = float(A.shape[1] * A.shape[2] - A.sum()) / A.sum() # share of positive weight edges

gnn = grap.GraphAutoencoder(nodes_n=X.shape[-2], feat_sz=X.shape[-1], activation=tf.nn.tanh)
gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
# gnn.fit(X, A, epochs=100, validation_data=(X,A), callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)])
gnn.fit((X, A_tilde), A, epochs=100, validation_data=((X,A_tilde),A), callbacks=[tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)])

A_hat = gnn((X, A_tilde))
