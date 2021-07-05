import numpy as np


def normalized_adjacency(A):

    A_tilde = A + np.matrix(np.eye(A.shape[0])) # add identity for self aggregation   
    D = np.array(np.sum(A_tilde, axis=1), dtype=np.float32) # compute outdegree (= rowsum)
    D = np.diag(np.power(D,-0.5).flatten()) # normalize (**-(1/2)) and diagonalize
    
    return np.matmul(D, np.matmul(A_tilde, D))    


def make_toy_graph():
    ''' make toy adjacency and feature matrix '''
    A = np.matrix([[0, 1, 0, 0, 0],
                   [1, 0, 1, 1, 0], 
                   [0, 1, 0, 0, 1],
                   [0, 1, 0, 0, 1],
                   [0, 0, 1, 1, 0]],
                    dtype=np.float32)
    A =  A.astype('float32')

    A_tilde = normalized_adjacency(A)

    X = np.matrix([[i/4, i/3, 1/i] for i in range(1, A.shape[0]+1)], dtype=np.float32)

    return X[np.newaxis,:,:], A[np.newaxis,:,:], A_tilde[np.newaxis,:,:] # add extra dim for batch, otheriwse model.fit does not work out of the box
