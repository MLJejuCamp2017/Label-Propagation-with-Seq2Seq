import numpy as np
import scipy as sp
import scipy.sparse
from sklearn.preprocessing import normalize
from time import time
import pickle

# similarity analysis using GPUs
import faiss

# load all data (vectors)
L = sp.sparse.load_npz('./data/graph/labeled.npz')
U = sp.sparse.load_npz('./data/graph/unlabeled.npz')
M = sp.sparse.vstack([L,U]) # combining labeled data with unlabeled data

# convert sparse matrix to dense matrix
# and change type from 'float64' to 'float32' since 'faiss' doesn't support 'float64' type
M = M.toarray()
M = M.astype('float32')
M = normalize(M) # L2 Norm before calculating cosine similarity

last_index_l = L.shape[0]
last_index_u = last_index_l + U.shape[0]

# we only keep the closest neighbors
max_neighs = 5
size = M.shape[0]

""" FAISS operations """
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatIP(res, M.shape[1]) # build the index

index.add(M) # add vectors to the index
""""""
def compute_graph_for_embedding(graph,edges_weights,edges_ll,edges_lu,edges_uu):
    batch_size = 1000
    batch_num = int(np.ceil(size / batch_size))

    sims, inds = [], []

    for i in tqdm(range(batch_num)):
        # actual search
        similarities, indices = index.search(M[i*batch_size:int(np.min([(i+1)*batch_size, size]))],max_neighs+1)

        # remove self-references
        batch_ids = np.vstack(np.arange(i*batch_size, int(np.min([(i+1)*batch_size, size]))))
        xs, ys = np.where(indices!=batch_ids)
        similarities[xs,ys] = 0

        sims.extend(similarities)
        inds.extend(indices)
    print()

    for i in tqdm(range(size)):
        neighbors_indices = list(inds[i][sims[i].argsort()[-max_neighs::][::-1]])
        correct_indices = [j for j in neighbors_indices if i < j]
        graph.update({i:correct_indices})

        n = len(correct_indices)

        if n > 0:
            edges = list(zip([i] * n, correct_indices))
            take_indices = [np.where(inds[i]==x)[0][0] for x in correct_indices]
            edges_weights.update(dict(zip(edges,np.take(sims[i],take_indices))))

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
    
    compute_graph_for_embedding(graph,edges_weights,edges_ll,edges_lu,edges_uu)
    
    # save to file the data structure
    pickle.dump(dict(graph), open("./data/graph/graph.p", "wb"))
    pickle.dump(dict(edges_weights), open("./data/graph/edges_weights.p", "wb"))
    pickle.dump(list(edges_ll), open("./data/graph/edges_ll.p", "wb"))
    pickle.dump(list(edges_lu), open("./data/graph/edges_lu.p", "wb"))
    pickle.dump(list(edges_uu), open("./data/graph/edges_uu.p", "wb"))