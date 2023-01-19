import numpy as np
import pandas as pd
import tqdm as tq


#### Data

#load
df_A = pd.read_csv('./data/MUTAG/MUTAG_A.txt',header=None)
df_B = pd.read_csv('./data/MUTAG/MUTAG_graph_indicator.txt',header=None)
df_C = pd.read_csv('./data/MUTAG/MUTAG_node_labels.txt',header=None)
#df_D = pd.read_csv('./data/MUTAG/MUTAG_graph_labels.txt',header=None)


node_num = len(df_C)
adj_total = np.zeros((node_num+1,node_num+1),dtype=int)
A = df_A.to_numpy(dtype=np.int32)
for row,col in A:
    adj_total[row-1][col-1] = 1


B = df_B.to_numpy(dtype=np.int32)
sub_graphs_class ={} # (graph_id : idx )
for idx,graph_id in enumerate(B):
    if graph_id[0] not in sub_graphs_class:
        sub_graphs_class[graph_id[0]] = [idx]

    else :
        sub_graphs_class[graph_id[0]] += [idx]


# adjacency matrix per class
class_num = len(sub_graphs_class)
sub_graphs_adj = {}
for i in range(1,class_num+1):
    start = sub_graphs_class[i][0]
    end = sub_graphs_class[i][-1]+1
    sub_graphs_adj[i] = adj_total[start:end,start:end]



def adj_to_degree(T):
    D = np.zeros((T.shape),dtype=np.int32)
    k = 0
    for i in T:
        D[k][k] = sum(i)
        k += 1
    return D

##feature map

#labeling
def initial_labeling(D): #input = degree matrix
    graph_label = {}
    for i in range(len(D)):
        graph_label[i] = str(D[i][i])
    return graph_label


def multi_labeling(G1,adj1):
    G = G1.copy()

    #labeling
    for x, j in enumerate(adj1):
        for y, k in enumerate(j):
            if k != 0:
                link = (x, y)

                if x == y:
                    continue
                else:
                    G[x] += G1[y]
    #sorting
    for i in G:
        G[i] = ''.join(sorted(G[i]))
    return G


def label_compression(G1,G0,H1,H0):
    G1 = G1.copy()
    H1 = H1.copy()
    J = list(G0.values()) + list(H0.values())
    criterion_scalar = max(map(int, J))
    hash_function = {'minima': criterion_scalar}

    domain = J
    for i in domain:
        if i in hash_function:
            continue
        else:
            hash_function[i] = max(hash_function.values()) + 1

    for i in G1:
        G1[i] = str(hash_function[G0[i]])
    for j in H1:
        H1[j] = str(hash_function[H0[j]])

    return G1,H1

'''
def is_end(G1,G0): # len(G1) == len(G2) True
    partitions0 = {}
    partitions1 = {}
    for i in range(len(G0)):
        if G0[i] in partitions0:
            partitions0[G0[i]] += 1
        else:
            partitions0[G0[i]] = 1

    for i in range(len(G1)):
        if G1[i] in partitions1:
            partitions1[G1[i]] += 1
        else:
            partitions1[G1[i]] = 1

    C0 = sorted(list(partitions0.values()))
    C1 = sorted(list(partitions1.values()))
    return C0 == C1
'''

#subtree kernel (counting labels)
def subtree_kernel(G):
    feature_vector = {}
    for i in range(len(G)):
        if G[i] in feature_vector:
            feature_vector[G[i]] += 1
        else:
            feature_vector[G[i]] = 1
    return feature_vector


def kernel_product(feature_G,feature_H):
    ans = 0
    for key in feature_G:
        if key in feature_H:
            ans += feature_G[key] * feature_H[key]
        else:
            ans += 0


    return ans


###########
sim_matrix = np.zeros((len(sub_graphs_class),len(sub_graphs_class)))
for p in tq.tqdm(range(1,len(sub_graphs_class)+1)):
    for q in range(p+1,len(sub_graphs_class)+1):

        adj1 = sub_graphs_adj[p]
        adj2 = sub_graphs_adj[q]

        graph1_ = adj_to_degree(adj1)
        graph2_ = adj_to_degree(adj2)

        graph1 = initial_labeling(graph1_)
        graph2 = initial_labeling(graph2_)

        similiarity = 0
        height = 3
        for i in range(height):
            a = subtree_kernel(graph1)
            b = subtree_kernel(graph2)
            similiarity += kernel_product(a,b)

            graph11 = multi_labeling(graph1,adj1)
            graph22 = multi_labeling(graph2,adj2)
            graph11,graph22 = label_compression(graph11,graph1,graph22,graph2)

            graph1 = graph11
            graph2 = graph22


        sim_matrix[p - 1][q - 1] = similiarity
        print('subgraph {}와 {}의 similiarity = {}'.format(p, q, similiarity))







































































































