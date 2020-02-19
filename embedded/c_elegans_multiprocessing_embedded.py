
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import statistics
import numpy as np
from c_elegans_multigraph_embedded import self_modeling
from multiprocessing import Pool
#import matplotlib.pyplot as plt

if __name__ == '__main__':
    start_time = time.time()

    # Number of experiments to be performed (cores to be used)
    number_experiments = 1

    # All neuromuscular junctions to be removed
    excl_file = "base/excluded_total"

    # The connectome file
    conn_file = "base/full_connectome_edgelist"

    # percentage of negative connections (not usedd in this implementation)
    perc_neg = 0

    clipp = 44
    learning_rate = 0.00001  # Hebbian learning rate
    tau_local = 18000  # No. of state updates before randomize
    set1 = 1000 # No. of reset convergence cycles before self-optimization
    set2 = 1000 # ... during self optimization
    set3 = 1000 # ... after self-optimization
    ss = set1+set2

    arguments=[(conn_file,excl_file,perc_neg,clipp,learning_rate,tau_local,set1,set2,set3)]*number_experiments
    p = Pool()
    result = p.starmap(self_modeling,arguments)

    p.close()
    p.join()

    relaxs = []

    relaxs_11 = []
    relaxs_12 = []
    relaxs_13 = []
    relaxs_21 = []
    relaxs_22 = []
    relaxs_hc1 = []
    relaxs_hc2 = []



    for tup in result:
        relaxs.append(tup[0][1:])


        relaxs_11.append(tup[4][1:])
        relaxs_12.append(tup[5][1:])
        relaxs_13.append(tup[6][1:])
        relaxs_21.append(tup[7][1:])
        relaxs_22.append(tup[8][1:])
        relaxs_hc1.append(tup[9][1:])
        relaxs_hc2.append(tup[10][1:])


    print("Local attractor values (Energy) general ")
    print(relaxs)
    print("Local attractor values (Energy) 11")
    print(relaxs_11)
    print("Local attractor values (Energy) 12")
    print(relaxs_12)
    print("Local attractor values (Energy) 13")
    print(relaxs_13)
    print("Local attractor values (Energy) 21")
    print(relaxs_21)
    print("Local attractor values (Energy) 22")
    print(relaxs_22)
    print("Local attractor values (Energy) hc1")
    print(relaxs_hc1)
    print("Local attractor values (Energy) hc2")
    print(relaxs_hc2)


    n_arr = np.array(relaxs)
    #print(n_arr)
    n_arr_t = np.transpose(n_arr)
    #print(n_arr_t)
    resu = np.average(n_arr_t,axis=1)
    print("Local attractor values averaged")
    print(*resu, sep=',\n')

    n_arr_11 = np.array(relaxs_11)
    n_arr_t_11 = np.transpose(n_arr_11)
    resu_11 = np.average(n_arr_t_11,axis=1)
    print("Local attractor values (Energy) Average cluster 11 ")
    print(*resu_11, sep=',\n')

    n_arr_12 = np.array(relaxs_12)
    n_arr_t_12 = np.transpose(n_arr_12)
    resu_12 = np.average(n_arr_t_12,axis=1)
    print("Local attractor values (Energy) Average cluster 12 ")
    print(*resu_12, sep=',\n')

    n_arr_13 = np.array(relaxs_13)
    n_arr_t_13 = np.transpose(n_arr_13)
    resu_13 = np.average(n_arr_t_13,axis=1)
    print("Local attractor values (Energy) Average cluster 13 ")
    print(*resu_13, sep=',\n')

    n_arr_21 = np.array(relaxs_21)
    n_arr_t_21 = np.transpose(n_arr_21)
    resu_21 = np.average(n_arr_t_21,axis=1)
    print("Local attractor values (Energy) Average cluster 21 ")
    print(*resu_21, sep=',\n')

    n_arr_22 = np.array(relaxs_22)
    n_arr_t_22 = np.transpose(n_arr_22)
    resu_22 = np.average(n_arr_t_22,axis=1)
    print("Local attractor values (Energy) Average cluster 22 ")
    print(*resu_22, sep=',\n')

    n_arr_hc1 = np.array(relaxs_hc1)
    n_arr_t_hc1 = np.transpose(n_arr_hc1)
    resu_hc1 = np.average(n_arr_t_hc1,axis=1)
    print("Local attractor values (Energy) Average cluster hc1 ")
    print(*resu_hc1, sep=',\n')

    n_arr_hc2 = np.array(relaxs_hc2)
    n_arr_t_hc2 = np.transpose(n_arr_hc2)
    resu_hc2 = np.average(n_arr_t_hc2,axis=1)
    print("Local attractor values (Energy) Average cluster hc2 ")
    print(*resu_hc2, sep=',\n')

    timing=(time.time() - start_time)
    print("-----Total time -----",timing)
