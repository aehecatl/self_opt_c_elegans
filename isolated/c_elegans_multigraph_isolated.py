#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The self-optimization in this work is mainly explored in the following works:

[1] Watson, R. A., Buckley, C. L., & Mills, R. (2011).
    Optimization in “self-modeling” complex adaptive systems. Complexity, 16(5), 17-26. doi:10.1002/cplx.20346

[2] Watson, R. A., Buckley, C. L., & Mills, R. (2011). Global Adaptation in Networks of Selfish Components: Emergent Associative Memory at the System Scale.
    Artificial Life, 17(3), 147-166. doi:10.1162/artl_a_00029

This code apply self-optimization in the C. elegans connectome as a directed multigraph.
'''


import re
import networkx as nx
import numpy as np

# Create discrete-time, discrete-state Hopfield Neural Network
# all states are set to zero
def create_network(connectome_file,file_excluded,perc,clipp):

    # Neurons we won't consider
    excluded = []

    with open(file_excluded,mode='r') as f:
        for line in f:
            a = re.split(",+",line)
            excluded.append(a[0])

    # The graph associated to this file (nodes are strings)
    R = nx.MultiDiGraph()

    # Now we read the file and fill R (taking out neuromucular junctions)
    with open(connectome_file,mode='r') as f:
        for line in f:
            a = re.split(",+",line)

            if not((a[0] in excluded) or (a[1] in excluded)):

                # Normalize weights clipping values greater than "clipp" to 1
                normalize = 0
                if float(a[2]) > clipp:
                    normalize = 1
                else:
                    normalize = float(a[2])/clipp

                # Add random negative connections
                if perc > 0:
                    s=np.random.uniform(0,1)
                    if(s<perc):
                        normalize=normalize*(-1)

                R.add_edge(a[0],a[1],weight=normalize,weight_original=normalize)

    size = R.number_of_nodes()
    edg = R.number_of_edges()

    G = nx.convert_node_labels_to_integers(R,ordering='sorted')

    # Adding extra zero connections (Hopfield network)

    for i in range(0, size, 1):
            for j in range(i, size, 1):
                if not((i,j) in G.edges()):
                    G.add_edge(i, j, weight=0, weight_original=0)

                if not((j,i) in G.edges()):
                   G.add_edge(j, i, weight=0, weight_original=0)

    return G,size,edg


# States are randomized in a discrete way {-1,1}
def randomize_states(G,size):
    np.random.seed()
    for i in range(size):
        if np.random.uniform(0,1) < 0.5:
            G.node[i]['state'] = -1
        else:
            G.node[i]['state'] = 1


# Changes 0 to 1 and viceversa
def inverse(i):
    if i==1:
        return -1
    else:
        return 1

# Asynchronous update (i.e. states are update one at time)
def update_states(G,size):
    ran_node = np.random.randint(size)
    actual_state = G.node[ran_node]['state']

    U_with_Actual = 0
    U_with_Change = 0
    suma = 0

    op_state = inverse(actual_state)


    for neigh in G.predecessors(ran_node):
        stn = G.node[neigh]['state']
        suma=0
        for reps in G[neigh][ran_node].keys():
            suma = suma + G[neigh][ran_node][reps]['weight']

        U_with_Actual = U_with_Actual + suma*actual_state*stn
        U_with_Change = U_with_Change + suma*op_state*stn

    if U_with_Change > U_with_Actual:
        G.node[ran_node]['state'] = op_state


# Threshold function used to constrain weights in [-1,1]
def threshold_function(argument):
    if argument < -1:
        return -1
    elif argument > 1:
        return 1
    else:
        return argument

# Hebbian learning rule applied at the end of the relaxation period
# (energy attractor with state updates)
def hebbian_learning(delta,G):
    for u, v, keys, weight in G.edges(data='weight', keys=True):
        if weight is not None:
            G[u][v][keys]['weight'] = threshold_function( G[u][v][keys]['weight'] +
                                          delta*G.node[u]['state']*G.node[v]['state'])
            pass

# The original energy function uses the weights that define the original constraint satisfaction problem.
# That mean, the original weights are not updated by Hebbian learning.
def energy_function(G):
    energy = 0
    for u, v, keys, weight in G.edges(data='weight_original', keys=True):
        if weight is not None:
            energy = energy + G[u][v][keys]['weight_original']*G.node[u]['state']*G.node[v]['state']
            pass
    return -energy

# Count the satisfied connections
def constraint_counter(G,n_edg):
    count = 0

    for u, v, keys, weight in G.edges(data='weight_original', keys=True):
        if weight is not None:
            cons = G[u][v][keys]['weight_original']*G.node[u]['state']*G.node[v]['state']
            if cons > 0:
                count = count+1
            pass
    return count*(100/n_edg)


def self_modeling(conn_file,file_excl,perc,clipp,delta,tau,set1,set2,set3):

    graph,size,n_edg = create_network(conn_file,file_excl,perc,clipp)
    randomize_states(graph,size)
    constraints_initial = []
    constraints_final = []
    constraints_final2 = []
    energy_function_states = []

    energy_function_states.append(energy_function(graph))

    for r in range(set1):
        for t in range(tau):
            update_states(graph,size)

        energy_function_states.append(energy_function(graph))
        constraints_initial.append(constraint_counter(graph,n_edg))
        # States are randomized such that the network can explored the state space properly
        randomize_states(graph,size)

    for r in range(set2):
        for t in range(tau):
            update_states(graph,size)

        energy_function_states.append(energy_function(graph))

        constraints_final.append(constraint_counter(graph,n_edg))

        # Learning is applied to reinforce attractors
        hebbian_learning(delta,graph)

        randomize_states(graph,size)

    for r in range(set3):
        for t in range(tau):
            update_states(graph,size)

        energy_function_states.append(energy_function(graph))
        constraints_final2.append(constraint_counter(graph,n_edg))

        randomize_states(graph,size)

    return energy_function_states,constraints_initial,constraints_final,constraints_final2


#if __name__ == '__main__':

    #G,size=create_network()
    #randomize_states(G,size)
