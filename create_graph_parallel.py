"""
This script generates the dictionaries that I need for representing 
the graph connecting the review embeddings, and to access them later in constant time.
A node is only connected only to its closest neighbors.

The job is split into three processes for a reasonable speedup

For convenience I also save the indices corresponding to the files in the order 
they appear in the glob function, so that we know what number corresponds to what review.
"""
import numpy as np
import scipy as sp
import scipy.sparse
from sklearn.neighbors import LSHForest
from time import time
import pickle
import glob

L = sp.sparse.load_npz('./data/graph/labeled.npz')
U = sp.sparse.load_npz('./data/graph/unlabeled.npz')
M = sp.sparse.vstack([L,U])
last_index_l = L.shape[0]
last_index_u = last_index_l + U.shape[0]

# we only keep the closest neighbors
max_neighs = 5
size = M.shape[0]

#lshf = LSHForest(n_estimators=15, n_candidates=50, n_neighbors=6, random_state=42)
lshf = LSHForest(random_state=42) 
lshf.fit(M)


def compute_graph_for_embedding(graph,edges_weights,edges_ll,edges_lu,edges_uu):
    """
    Function for computing the subgraph for nodes.
    Note that the edges_* structures are meant to be used later in the objective function of the Conv-NN
    and are not of any particular interest for the sake of the graph creation
    :param graph: a dictionary mapping a node to a list of neighbors
    :param edges_weights: dict that maps an egde (u,v) to its weight, in our case cosine_similarity
    :param edges_ll: edges from labeled node to labeled node
    :param edges_lu: edges from labeled node to unlabeled node
    :param edges_uu: edges from unlabeled node to unlabeled node
    """
    batch_size = 1000
    batch_num = int(np.ceil(size / batch_size))
    
    sims, inds = [], []

    for i in range(batch_num):
        t_str = time()
        distances, indices = lshf.kneighbors(M[i*batch_size:(i+1)*batch_size],\
                                            n_neighbors=6)
        batch_ids = np.vstack(np.arange(i*batch_size, int(np.min([(i+1)*batch_size, size]))))
        xs, ys = np.where(indices==batch_ids)
        distances[xs,ys] = 2.0
        sims.extend(1-distances)
        inds.extend(inds)
        print(i, time() - t_str, end='\r')
    print()
    pickle.dump([sims, inds], open("./data/graph/approx_nn.p", "wb"))
    #[sims, inds] = pickle.load(open("./data/graph/approx_nn.p", "rb"))

    for i in range(size):
        neighbors_indices = list(indices[i][sims[i].argsort()[-max_neighs::][::-1]])
        correct_indices = [j for j in neighbors_indices if i < j]

        graph.update({i:correct_indices})

        n = len(correct_indices)

        if n > 0:
            edges = list(zip([i] * n, correct_indices))
            edges_weights.update(dict(zip(edges,np.take(sims[i],correct_indices))))

            for j in correct_indices:
                if (0 <= i < last_index_l) and (0 <= j < last_index_l):
                    edges_ll.append((i,j))
                elif (0 <= i < last_index_l) and (last_index_l <= j < last_index_u):
                    edges_lu.append((i,j))
                else:
                    edges_uu.append((i,j))
    return

if __name__ == '__main__':

    graph = dict()
    edges_weights = dict()
    edges_ll = list()
    edges_lu = list()
    edges_uu = list()

    compute_graph_for_embedding(graph, edges_weights, edges_ll, edges_lu, edges_uu)

    # save to file the data structure that we worked so hard to compute
    pickle.dump(dict(graph), open("./data/graph/graph.p", "wb"))
    pickle.dump(dict(edges_weights), open("./data/graph/edges_weights.p", "wb"))
    pickle.dump(list(edges_ll), open("./data/graph/edges_ll.p", "wb"))
    pickle.dump(list(edges_lu), open("./data/graph/edges_lu.p", "wb"))
    pickle.dump(list(edges_uu), open("./data/graph/edges_uu.p", "wb"))
