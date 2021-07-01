import numpy


def make_toy_graph():
    ''' make toy adjacency and feature matrix '''
    A = np.matrix([[0, 1, 0, 0],
                    [0, 0, 1, 1], 
                    [0, 1, 0, 0],
                    [1, 0, 1, 0]],
                    dtype=float)
    A = A + np.matrix(np.eye(A.shape[0])) # add identity for self aggregation   
    D = np.array(np.sum(A, axis=0))[0]
    D = np.matrix(np.diag(D))
    A = np.linalg.inv(D) * A

    X = np.matrix([[i, -i]
            for i in range(A.shape[0])
            ], dtype=float)
    return X, A



# get toy data
X, A = make_toy_graph()

