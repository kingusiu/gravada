import tensorflow as tf

import data.input_samples as inpu
import models.graph_nn as grap



# get toy data
X, A = inpu.make_toy_graph()

gnn = grap.GraphAutoencoder(nodes_n=X.shape[0], feat_sz=X.shape[1], activation=tf.nn.elu)
