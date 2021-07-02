import unittest
import numpy as np
import tensorflow as tf
import models.layers as lays


class GraphConvLayerTest(unittest.TestCase):

    def test_graph_conv_layer_forward(self):
        # test 2-dim feature output graph layer
        graph_layer = lays.GraphConvolution(output_sz=2, activation=tf.keras.activations.linear)

        # inputs
        X = np.matrix([[ 1.        , -1.        ,  1.        ],
                        [ 2.        , -2.        ,  0.5       ],
                        [ 3.        , -3.        ,  0.33333334],
                        [ 4.        , -4.        ,  0.25      ]], dtype=float32)

        A = np.matrix([[0.5       , 0.5       , 0.        , 0.        ],
                        [0.        , 0.33333334, 0.33333334, 0.33333334],
                        [0.        , 0.33333334, 0.33333334, 0.        ],
                        [0.5       , 0.        , 0.5       , 0.5       ]], dtype=float32)

        X_pred = graph_layer(X, A)

        W = graph_layer.kernel.numpy()

        # test that graph_layer(X, A) = A*X*W for identity activation
        X_pred_comp = A*X*W

        self.assertIsNone(np.testing.assert_allclose(X_pred, X_pred_comp))

if __name__ == '__main__':
    unittest.main()
