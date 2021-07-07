import numpy as np
import networkx as nx


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


def make_mask(idx, shape):
    mask = np.zeros(shape)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_karate_graph():
    club_dict = {'Mr. Hi' : 0, 
             'Officer': 1}

    G = nx.karate_club_graph()
    nodes_n = G.number_of_nodes()
    X = np.eye(nodes_n, dtype=np.float32) # featureless graph
    A = nx.to_numpy_array(G)
    A_tilde = normalized_adjacency(A)
    club_labels = nx.get_node_attributes(G,'club')
    y = np.array([club_dict[label] for label in club_labels.values()])
    return X, A, A_tilde, y

def make_karate_data_classifier():

    X, A, A_tilde, y = get_karate_graph()
    nodes_n = len(y)

    train_mask = make_mask(np.random.randint(nodes_n, size=int(nodes_n/3)), nodes_n) # third of nodes = training
    valid_mask = make_mask(np.random.randint(nodes_n, size=int(nodes_n/3)), nodes_n) # third of nodes = validation

    return X, A_tilde, y, train_mask, valid_mask


def make_karate_data_autoencoder(): 

    X, A, A_tilde, y = get_karate_graph()
    nodes_n = len(y)

    train_mask = np.random.choice([True, False], (nodes_n, nodes_n))
    valid_mask = np.random.choice([True, False], (nodes_n, nodes_n))

    return X, A_tilde, A, train_mask, valid_mask
 