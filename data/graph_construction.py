import numpy as np


def normalized_adjacency(A):
    D = np.array(np.sum(A, axis=2), dtype=np.float32) # compute outdegree (= rowsum)
    D = np.nan_to_num(np.power(D,-0.5), posinf=0, neginf=0) # normalize (**-(1/2))
    D = np.asarray([np.diagflat(dd) for dd in D]) # and diagonalize
    return np.matmul(D, np.matmul(A, D))

def make_adjacencies(particles, pt_idx=0):
    real_p_mask = particles[:,:,pt_idx] > 0
    adjacencies = (real_p_mask[:,:,np.newaxis] * real_p_mask[:,np.newaxis,:]).astype('float32')
    return adjacencies


def normalized_adjacency_no_selfref(A):

    A_tilde = A + np.matrix(np.eye(A.shape[0])) # add identity for self aggregation   
    D = np.array(np.sum(A_tilde, axis=1), dtype=np.float32) # compute outdegree (= rowsum)
    D = np.diag(np.power(D,-0.5).flatten()) # normalize (**-(1/2)) and diagonalize
    
    return np.matmul(D, np.matmul(A_tilde, D))    
