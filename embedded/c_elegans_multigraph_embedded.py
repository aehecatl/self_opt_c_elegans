# -*- coding: utf-8 -*-

'''
The self-optimization in this work is mainly explored in the following works:

[1] Watson, R. A., Buckley, C. L., & Mills, R. (2011).
    Optimization in “self-modeling” complex adaptive systems. Complexity, 16(5), 17-26. doi:10.1002/cplx.20346

[2] Watson, R. A., Buckley, C. L., & Mills, R. (2011). Global Adaptation in Networks of Selfish Components: Emergent Associative Memory at the System Scale.
    Artificial Life, 17(3), 147-166. doi:10.1162/artl_a_00029


This version has the procedure of [2] for updates
and the procedure of [1] for the Hebbian learning.
But apparently, the procedure for updating is
equivalent in [1] and [2]
'''

import re
import statistics
import networkx as nx
import numpy as np
from random import sample



# Create discrete-time, discrete-state Hopfield Neural Network
# all states are set to zero
# connectome_file contains all connections between neurons and muscles
def create_network(connectome_file,file_excluded,perc,clipp):

    # Neurons we won't consider
    excluded = []

    with open(file_excluded,mode='r') as f:
        for line in f:
            a = re.split(",+",line)
            excluded.append(a[0])

    # The graph associated to this file (nodes are strings)
    R = nx.MultiDiGraph()

    # Now we read the file and fill R
    with open(connectome_file,mode='r') as f:
        for line in f:
            a = re.split(",+",line)

            if not((a[0] in excluded) or (a[1] in excluded)):

                normalize = 0
                if float(a[2]) > clipp:
                    normalize = 1
                else:
                    normalize = float(a[2])/clipp

                R.add_edge(a[0],a[1],weight=normalize,weight_original=normalize,cs=0)

    size = R.number_of_nodes()
    edg = R.number_of_edges()

    #Map values to control clusters
    mapping = {'ADAL':0,'ADAR':1,'ADFL':2,'ADFR':3,'ADLL':4,'ADLR':5,'AFDL':6,'AFDR':7,'AIAL':8,'AIAR':9,'AIBL':10,'AIBR':11,'AIML':12,'AIMR':13,'AINL':14,'AINR':15,'AIYL':16,'AIYR':17,'AIZL':18,'AIZR':19,'ALNL':20,'ALNR':21,'ASEL':22,'ASER':23,'ASGL':24,'ASGR':25,'ASHL':26,'ASHR':27,'ASIL':28,'ASIR':29,'ASJL':30,'ASJR':31,'ASKL':32,'ASKR':33,'AWAL':34,'AWAR':35,'AWBL':36,'AWBR':37,'AWCL':38,'AWCR':39,'HSNL':40,'HSNR':41,'PLNL':42,'PLNR':43,'PVQL':44,'PVQR':45,'RIML':46,'RIMR':47,'RIR':48,'SAADL':49,'SAADR':50,'SAAVL':51,'SAAVR':52,'SMBDL':53,'SMBDR':54,'SMBVL':55,'SMBVR':56,'ALA':57,'AUAL':58,'AUAR':59,'AVEL':60,'AVER':61,'BAGL':62,'BAGR':63,'CEPDL':64,'CEPDR':65,'CEPVL':66,'CEPVR':67,'IL1DL':68,'IL1DR':69,'IL1L':70,'IL1R':71,'IL1VL':72,'IL1VR':73,'IL2DL':74,'IL2DR':75,'IL2L':76,'IL2R':77,'IL2VL':78,'IL2VR':79,'OLLL':80,'OLLR':81,'OLQDL':82,'OLQDR':83,'OLQVL':84,'OLQVR':85,'RIAL':86,'RIAR':87,'RIBL':88,'RIBR':89,'RICL':90,'RICR':91,'RIH':92,'RIPL':93,'RIPR':94,'RIS':95,'RIVL':96,'RIVR':97,'RMDDL':98,'RMDDR':99,'RMDL':100,'RMDR':101,'RMDVL':102,'RMDVR':103,'RMED':104,'RMEL':105,'RMER':106,'RMEV':107,'RMGL':108,'RMGR':109,'RMHL':110,'RMHR':111,'SIADL':112,'SIADR':113,'SIAVL':114,'SIAVR':115,'SIBDL':116,'SIBDR':117,'SIBVL':118,'SIBVR':119,'SMDDL':120,'SMDDR':121,'SMDVL':122,'SMDVR':123,'URADL':124,'URADR':125,'URAVL':126,'URAVR':127,'URBL':128,'URBR':129,'URXL':130,'URXR':131,'URYDL':132,'URYDR':133,'URYVL':134,'URYVR':135,'ADEL':136,'ADER':137,'AQR':138,'AVKL':139,'AVKR':140,'AVL':141,'DVC':142,'PVPL':143,'PVPR':144,'PVT':145,'RIGL':146,'RIGR':147,'RMFL':148,'RMFR':149,'ALML':150,'ALMR':151,'AS01':152,'AS07':153,'AS08':154,'AS09':155,'AS10':156,'AS11':157,'AVAL':158,'AVAR':159,'AVBL':160,'AVBR':161,'AVDL':162,'AVDR':163,'AVG':164,'AVJL':165,'AVJR':166,'AVM':167,'BDUL':168,'BDUR':169,'DA01':170,'DA06':171,'DA07':172,'DA08':173,'DA09':174,'DB05':175,'DB06':176,'DB07':177,'DD06':178,'DVA':179,'DVB':180,'FLPL':181,'FLPR':182,'LUAL':183,'LUAR':184,'PDA':185,'PDB':186,'PDEL':187,'PDER':188,'PHAL':189,'PHAR':190,'PHBL':191,'PHBR':192,'PHCL':193,'PHCR':194,'PLML':195,'PLMR':196,'PQR':197,'PVCL':198,'PVCR':199,'PVDL':200,'PVDR':201,'PVM':202,'PVNL':203,'PVNR':204,'PVR':205,'PVWL':206,'PVWR':207,'RID':208,'RIFL':209,'RIFR':210,'SABVL':211,'SABVR':212,'SDQL':213,'SDQR':214,'VA10':215,'VA11':216,'VA12':217,'VB07':218,'VB10':219,'VB11':220,'VD11':221,'VD12':222,'VD13':223,'AS02':224,'AS03':225,'AS04':226,'AS05':227,'AS06':228,'AVFL':229,'AVFR':230,'AVHL':231,'AVHR':232,'DA02':233,'DA03':234,'DA04':235,'DA05':236,'DB01':237,'DB02':238,'DB03':239,'DB04':240,'DD01':241,'DD02':242,'DD03':243,'DD04':244,'DD05':245,'SABD':246,'VA01':247,'VA02':248,'VA03':249,'VA04':250,'VA05':251,'VA06':252,'VA07':253,'VA08':254,'VA09':255,'VB01':256,'VB02':257,'VB03':258,'VB04':259,'VB05':260,'VB06':261,'VB08':262,'VB09':263,'VC01':264,'VC02':265,'VC03':266,'VC04':267,'VC05':268,'VD01':269,'VD02':270,'VD03':271,'VD04':272,'VD05':273,'VD06':274,'VD07':275,'VD08':276,'VD09':277,'VD10':278}

    #G = nx.convert_node_labels_to_integers(R,ordering='sorted')
    G = nx.relabel_nodes(R,mapping)

    visited = []
    visits = 0

    while visits < 1676:
        edge = sample(list(G.edges(keys=True)),1)
        if not edge in visited:
            if edge[0][0] in range(0,57) and edge[0][1] in range(57,279):
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight']
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original']
                visited.append(edge)
                visits = visits+1
            elif edge[0][0] in range(57,136) and (edge[0][1] in range(0,57) or edge[0][1] in range(136,279)):
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight']
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original']
                visited.append(edge)
                visits = visits+1
            elif edge[0][0] in range(136,150) and (edge[0][1] in range(0,136) or edge[0][1] in range(150,279)):
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight']
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original']
                visited.append(edge)
                visits = visits+1
            elif edge[0][0] in range(150,224) and (edge[0][1] in range(0,150) or edge[0][1] in range(224,279)):
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight']
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original']
                visited.append(edge)
                visits = visits+1
            elif edge[0][0] in range(224,279) and edge[0][1] in range(0,224):
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight']
                G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original'] = (-1)*G[edge[0][0]][edge[0][1]][edge[0][2]]['weight_original']
                visited.append(edge)
                visits = visits+1
            else:
                continue


    # The Hopfield network (fully connected network)
    # The self-loops will be removed in order to reduce the number of edges

    for i in range(0, size, 1):
            for j in range(i, size, 1):
                if not((i,j) in G.edges()):
                    G.add_edge(i, j, weight=0, weight_original=0,cs=10)

                if not((j,i) in G.edges()):
                   G.add_edge(j, i, weight=0, weight_original=0,cs=10)

    #print(G.number_of_nodes())
    return G,size,edg


# States are randomized in a discrete way {-1,1}
# In other works states can be continuous (e.g. [5] and [6])
def randomize_states(G,size):
    np.random.seed()
    for i in range(size):
        if np.random.uniform(0,1) < 0.5:
            G.node[i]['state'] = -1
        else:
            G.node[i]['state'] = 1

    #print("termine de aleatorizar")

# Changes 0 to 1 and viceversa
def inverse(i):
    if i==1:
        return -1
    else:
        return 1

# Asynchronous update (i.e. states are update one at time)
def update_states(G,size):
    ran_node = np.random.randint(size)   #np.random.choice(self.size, 1, replace=False)
    actual_state = G.node[ran_node]['state']

    U_with_Actual = 0
    U_with_Change = 0
    suma = 0

    op_state = inverse(actual_state)


    for neigh in G.predecessors(ran_node):
        #print(neigh)
        stn = G.node[neigh]['state']
        suma=0
        for reps in G[neigh][ran_node].keys():
            suma = suma + G[neigh][ran_node][reps]['weight']

        #print(suma)
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
# The learning rate, delta, can be changed to applied learning during relaxation
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

def energy_function_subset(G,first,last):
    energy = 0
    for u, v, keys, weight in G.subgraph([x for x in range(first,last)]).edges(data='weight_original', keys=True):
        if weight is not None:
            energy = energy + G[u][v][keys]['weight_original']*G.node[u]['state']*G.node[v]['state']
            pass
    return -energy

def constraint_counter(G,n_edg):
    count = 0

    for u, v, keys, weight in G.edges(data='weight_original', keys=True):
        if weight is not None:
            cons = G[u][v][keys]['weight_original']*G.node[u]['state']*G.node[v]['state']
            if cons > 0:
                #descomentar si se van a sacar graficas
                #G[u][v][keys]['cs'] = 1
                count = count+1
            pass
    return count*(100/n_edg)

def constraint_counter_subset(G,first,last):
    count = 0
    count2 = 0

    for u, v, keys, weight in G.subgraph([x for x in range(first,last)]).edges(data='weight_original', keys=True):
        if weight is not None:
            cons = G[u][v][keys]['weight_original']*G.node[u]['state']*G.node[v]['state']
            if cons > 0:
                #descomentar si se van a sacar graficas
                #G[u][v][keys]['cs'] = 1
                count = count+1
                count2 = count2+1
            elif cons < 0:
                count2 = count2+1
            pass
    return count*(100/count2)

# delta: learning rate
# tau: steps to reach an atractor
# relaxations: how many times the network is allowed to reach an attractor
def self_modeling(conn_file,file_excl,perc,clipp,delta,tau,set1,set2,set3):

    graph,size,n_edg = create_network(conn_file,file_excl,perc,clipp)

    randomize_states(graph,size)
    # Descomentar cuando se ejecuta una vez para obtener gráfico
    #nx.write_edgelist(graph,"bl",data=['weight_original','cs'])
    constraints_initial = []
    constraints_final = []
    constraints_final2 = []
    energy_function_states = []

    energy_function_states_11 = []
    energy_function_states_12 = []
    energy_function_states_13 = []
    energy_function_states_21 = []
    energy_function_states_22 = []
    energy_function_states_hc1 = []
    energy_function_states_hc2 = []

    constraints_initial_11 = []
    constraints_initial_12 = []
    constraints_initial_13 = []
    constraints_initial_21 = []
    constraints_initial_22 = []
    constraints_initial_hc1 = []
    constraints_initial_hc2 = []

    constraints_final_11 = []
    constraints_final_12 = []
    constraints_final_13 = []
    constraints_final_21 = []
    constraints_final_22 = []
    constraints_final_hc1 = []
    constraints_final_hc2 = []

    constraints_final2_11 = []
    constraints_final2_12 = []
    constraints_final2_13 = []
    constraints_final2_21 = []
    constraints_final2_22 = []
    constraints_final2_hc1 = []
    constraints_final2_hc2 = []

    energy_function_states.append(energy_function(graph))

    for r in range(set1):
            # Reaching an attractor
        for t in range(tau):
            update_states(graph,size)

        # The energy at the end of the process should be lower than the energy at the beginning
        energy_function_states.append(energy_function(graph))
        constraints_initial.append(constraint_counter(graph,n_edg))

        energy_function_states_11.append(energy_function_subset(graph,0,57))
        energy_function_states_12.append(energy_function_subset(graph,57,136))
        energy_function_states_13.append(energy_function_subset(graph,136,150))
        energy_function_states_21.append(energy_function_subset(graph,150,224))
        energy_function_states_22.append(energy_function_subset(graph,224,279))
        energy_function_states_hc1.append(energy_function_subset(graph,0,150))
        energy_function_states_hc2.append(energy_function_subset(graph,150,279))

        constraints_initial_11.append(constraint_counter_subset(graph,0,57))
        constraints_initial_12.append(constraint_counter_subset(graph,57,136))
        constraints_initial_13.append(constraint_counter_subset(graph,136,150))
        constraints_initial_21.append(constraint_counter_subset(graph,150,224))
        constraints_initial_22.append(constraint_counter_subset(graph,224,279))
        constraints_initial_hc1.append(constraint_counter_subset(graph,0,150))
        constraints_initial_hc2.append(constraint_counter_subset(graph,150,279))

        # States are randomized such that the network can explored the state space properly
        randomize_states(graph,size)

    # Descomentar cuando se ejecuta una vez para obtener gráfico
    #nx.write_edgelist(graph,"ml",data=['weight_original','cs'])


    for r in range(set2):
            # Reaching an attractor
        for t in range(tau):
            update_states(graph,size)

            # If the energy is measured here, it does not always decrease
            #energy_function_states.append(energy_function(graph))

        # The energy at the end of the process should be lower than the energy at the beginning
        energy_function_states.append(energy_function(graph))

        constraints_final.append(constraint_counter(graph,n_edg))

        energy_function_states_11.append(energy_function_subset(graph,0,57))
        energy_function_states_12.append(energy_function_subset(graph,57,136))
        energy_function_states_13.append(energy_function_subset(graph,136,150))
        energy_function_states_21.append(energy_function_subset(graph,150,224))
        energy_function_states_22.append(energy_function_subset(graph,224,279))
        energy_function_states_hc1.append(energy_function_subset(graph,0,150))
        energy_function_states_hc2.append(energy_function_subset(graph,150,279))

        constraints_final_11.append(constraint_counter_subset(graph,0,57))
        constraints_final_12.append(constraint_counter_subset(graph,57,136))
        constraints_final_13.append(constraint_counter_subset(graph,136,150))
        constraints_final_21.append(constraint_counter_subset(graph,150,224))
        constraints_final_22.append(constraint_counter_subset(graph,224,279))
        constraints_final_hc1.append(constraint_counter_subset(graph,0,150))
        constraints_final_hc2.append(constraint_counter_subset(graph,150,279))

        #print(energy_function_states[-1])
        # Learning is applied to reinforce attractors
        hebbian_learning(delta,graph)

        # States are randomized such that the network can explored the state space properly
        randomize_states(graph,size)

    #Descomentar cuando ejecuta una vez para obtener gŕafico
    #nx.write_edgelist(graph,"al",data=['weight_original','cs'])


    for r in range(set3):
            # Reaching an attractor
        for t in range(tau):
            update_states(graph,size)

        # The energy at the end of the process should be lower than the energy $
        energy_function_states.append(energy_function(graph))
        constraints_final2.append(constraint_counter(graph,n_edg))

        energy_function_states_11.append(energy_function_subset(graph,0,57))
        energy_function_states_12.append(energy_function_subset(graph,57,136))
        energy_function_states_13.append(energy_function_subset(graph,136,150))
        energy_function_states_21.append(energy_function_subset(graph,150,224))
        energy_function_states_22.append(energy_function_subset(graph,224,279))
        energy_function_states_hc1.append(energy_function_subset(graph,0,150))
        energy_function_states_hc2.append(energy_function_subset(graph,150,279))

        constraints_final2_11.append(constraint_counter_subset(graph,0,57))
        constraints_final2_12.append(constraint_counter_subset(graph,57,136))
        constraints_final2_13.append(constraint_counter_subset(graph,136,150))
        constraints_final2_21.append(constraint_counter_subset(graph,150,224))
        constraints_final2_22.append(constraint_counter_subset(graph,224,279))
        constraints_final2_hc1.append(constraint_counter_subset(graph,0,150))
        constraints_final2_hc2.append(constraint_counter_subset(graph,150,279))

        # States are randomized such that the network can explored the state sp$
        randomize_states(graph,size)


    return energy_function_states,constraints_initial,constraints_final,constraints_final2,energy_function_states_11,energy_function_states_12,energy_function_states_13,energy_function_states_21,energy_function_states_22,energy_function_states_hc1,energy_function_states_hc2,constraints_initial_11,constraints_initial_12,constraints_initial_13,constraints_initial_21,constraints_initial_22,constraints_initial_hc1,constraints_initial_hc2,constraints_final_11,constraints_final_12,constraints_final_13,constraints_final_21,constraints_final_22,constraints_final_hc1,constraints_final_hc2,constraints_final2_11,constraints_final2_12,constraints_final2_13,constraints_final2_21,constraints_final2_22,constraints_final2_hc1,constraints_final2_hc2


#if __name__ == '__main__':

    #G,size,edg=create_network()
    #randomize_states(G,size)
    #result = constraint_counter_subset(G,,57)
    #result = self_modeling(0.00001,4000, 100)

    #print(result)
