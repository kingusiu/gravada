import numpy as np

def make_toy_graph():
    ''' make toy adjacency and feature matrix '''
    A = np.matrix([[0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0], 
                    [0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 1, 1, 0, 0]],
                    dtype=np.float32)
    A = A + np.matrix(np.eye(A.shape[0])) # add identity for self aggregation   
    D = np.array(np.sum(A, axis=0))[0]
    D = np.matrix(np.diag(D), dtype=np.float32)
    A = np.linalg.inv(D) * A

    X = np.matrix([[i, -i, 1/i, i*10] for i in range(1, A.shape[0]+1)], dtype=np.float32)
    
    return X, A.astype('float32')
