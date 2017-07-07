import numpy as np
import networkx as nx
from embeddings_graph import EmbeddingsGraph
from bytenet import ByteNet
from data_koen import KOEN
import unicodedata

graph = EmbeddingsGraph().graph
batch_size = 32
data = KOEN(batch_size, 'train')
data2 = KOEN(batch_size, 'train.mono')

with open('./data/raw/ko.train', 'r') as f:
    ss_L = f.readlines()
    ss_L = [unicodedata.normalize("NFKD", unicode_str[:-1]) for unicode_str in ss_L]
    
with open('./data/raw/ko.train.mono', 'r') as f:
    ss_U = f.readlines()
    ss_U = [unicodedata.normalize("NFKD", unicode_str[:-1]) for unicode_str in ss_U]

l = len(ss_L) #last index of labeled samples
u = l + len(ss_U) #last index of all samples

for key, value in data2.ids.items():
    data.ids[key+l] = data.num_data + value
    
data.source.extend(data2.source)

##################
##### remove #####
##################
for (u, v) in graph.edges():
    try:
        data.ids[u]
        data.ids[v]
    except:
        graph.remove_edge(u,v)
        
data.source = np.array(data.source)
data.target = np.array(data.target)

def label(i):
    if 0 <= i < l:
        return data.target[data.ids[i]]


def next_batch(h_edges, start, finish):
    """
    Helper function for the iterator, note that the neural graph machines,
    due to its unique loss function, requires carefully crafted inputs

    Refer to the Neural Graph Machines paper, section 3 and 3.3 for more details
    """
    edges_ll = list()
    edges_lu = list()
    edges_uu = list()
    weights_ll = list()
    weights_lu = list()
    weights_uu = list()
    batch_edges = h_edges[start:finish]
    batch_edges = np.asarray(batch_edges)

    for i, j in batch_edges[:]:
        if (0 <= i < l) and (0 <= j < l):
            edges_ll.append((i, j))
            weights_ll.append(graph.get_edge_data(i,j)['weight'])
        elif (0 <= i < l) and (l <= j < u):
            edges_lu.append((i, j))
            weights_lu.append(graph.get_edge_data(i,j)['weight'])
        else:
            edges_uu.append((i, j))
            weights_uu.append(graph.get_edge_data(i,j)['weight'])

    u_ll = [e[0] for e in edges_ll]

    # number of incident edges for nodes u
    c_ull = [1 / len(graph.edges(n)) for n in u_ll]
    v_ll = [e[1] for e in edges_ll]
    c_vll = [1 / len(graph.edges(n)) for n in v_ll]
    nodes_ll_u = data.source[[data.ids[x] for x in u_ll]]

    labels_ll_u = np.vstack([label(n) for n in u_ll])

    nodes_ll_v = data.source[[data.ids[x] for x in v_ll]]

    labels_ll_v = np.vstack([label(n) for n in v_ll])

    u_lu = [e[0] for e in edges_lu]
    c_ulu = [1 / len(graph.edges(n)) for n in u_lu]
    nodes_lu_u = data.source[[data.ids[x] for x in u_lu]]
    nodes_lu_v = data.source[[data.ids[x] for x in [e[1] for e in edges_lu]]]

    labels_lu = np.vstack([label(n) for n in u_lu])

    nodes_uu_u = data.source[[data.ids[x] for x in [e[0] for e in edges_uu]]]
    nodes_uu_v = data.source[[data.ids[x] for x in [e[1] for e in edges_uu]]]

    return nodes_ll_u, nodes_ll_v, labels_ll_u, labels_ll_v, \
           nodes_uu_u, nodes_uu_v, nodes_lu_u, nodes_lu_v, \
           labels_lu, weights_ll, weights_lu, weights_uu, \
           c_ull, c_vll, c_ulu


def batch_iter(batch_size):
    """
        Generates a batch iterator for the dataset.
    """

    data_size = len(graph.edges())

    edges = np.random.permutation(graph.edges())

    num_batches = int(data_size / batch_size)

    if data_size % batch_size > 0:
        num_batches = int(data_size / batch_size) + 1

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield next_batch(edges,start_index,end_index)