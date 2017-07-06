#from tqdm import tqdm
#import numpy as np
#import scipy as sp
#import scipy.sparse
#from sklearn.neighbors import LSHForest
#from time import time
#import pickle
#import glob
#
#L = sp.sparse.load_npz('./data/graph/labeled.npz')
#U = sp.sparse.load_npz('./data/graph/unlabeled.npz')
#M = sp.sparse.vstack([L,U])
#last_index_l = L.shape[0]
#last_index_u = last_index_l + U.shape[0]
#
## we only keep the closest neighbors
#max_neighs = 5
#size = M.shape[0]
#
##lshf = LSHForest(n_estimators=15, n_candidates=50, n_neighbors=6, random_state=42)
#lshf = LSHForest(random_state=42) 
#lshf.fit(M)
#
#
#graph = dict()
#edges_weights = dict()
#edges_ll = list()
#edges_lu = list()
#edges_uu = list()
#
#batch_size = 1000
#batch_num = int(np.ceil(size / batch_size))
#
#sims, inds = [], []

#for i in range(batch_num):
#    t_str = time()
#    distances, indices = lshf.kneighbors(M[i*batch_size:int(np.min([(i+1)*batch_size, size]))],\
#                                        n_neighbors=6)
#    batch_ids = np.vstack(np.arange(i*batch_size, int(np.min([(i+1)*batch_size, size]))))
#    xs, ys = np.where(indices==batch_ids)
#    distances[xs,ys] = 2.0
#    sims.extend(1-distances)
#    inds.extend(inds)
#    print(i, time() - t_str, end='\r')
#print()
#pickle.dump([sims, inds], open("./data/graph/approx_nn.p", "wb"))
##[sims, inds] = pickle.load(open("./data/graph/approx_nn.p", "rb"))

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
#
#    # save to file the data structure that we worked so hard to compute
#    pickle.dump(dict(graph), open("./data/graph/graph.p", "wb"))
#    pickle.dump(dict(edges_weights), open("./data/graph/edges_weights.p", "wb"))
#    pickle.dump(list(edges_ll), open("./data/graph/edges_ll.p", "wb"))
#    pickle.dump(list(edges_lu), open("./data/graph/edges_lu.p", "wb"))
#    pickle.dump(list(edges_uu), open("./data/graph/edges_uu.p", "wb"))
