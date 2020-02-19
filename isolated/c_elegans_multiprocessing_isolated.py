
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Self-optimization in a directed multigraph with the posibility of
perform several simulations in parallel.
'''

import time
import statistics
import numpy as np
from c_elegans_multigraph_isolated import self_modeling
from multiprocessing import Pool
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start_time = time.time()

    # Number of experiments to be performed (cores to be used)
    number_experiments = 1

    # The corresponding cluster file
    # Load excluded_for_11 to run self-optimization with cluster 11,
    # excluded_for_12 to run self-optimization with cluster 12, and so on.
    excl_file = "base/excluded_for_11"

    # The connectome file
    conn_file = "base/full_connectome_edgelist"

    #Percentage of random negative connections in the network [0,1]
    perc_neg = 0.3

    clipp = 44
    learning_rate = 0.00001  # Hebbian learning rate
    tau_local = 8000  # No. of state updates before randomize
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

    last_values_cs_in = []
    last_values_cs_fin = []

    const_in = []
    const_fin = []
    const_fin2 = []

    prom_in = []
    prom_fin = []
    prom_fin2 = []
    sd_in = []
    sd_fin = []
    sd_fin2 = []


    for tup in result:

        relaxs.append(tup[0][1:])

        last_values_cs_in.append(tup[1][-1])
        last_values_cs_fin.append(tup[2][-1])
        const_in.append(tup[1])
        const_fin.append(tup[2])
        const_fin2.append(tup[3])
        prom_in.append(statistics.mean(tup[1]))
        prom_fin.append(statistics.mean(tup[2]))
        prom_fin2.append(statistics.mean(tup[3]))
        sd_in.append(statistics.stdev(tup[1]))
        sd_fin.append(statistics.stdev(tup[2]))
        sd_fin2.append(statistics.stdev(tup[3]))



    print("Local attractor values (Energy)")
    print(relaxs)

    print("-----------------------------------***************")
    print("Constraint satisfacion before self-optimization (1-%s)" % set1)
    print("Last value constraint satisfaction without Hebbian Learning")
    print(last_values_cs_in)
    print("Average")
    print(prom_in)
    print("Average under number simulations")
    print(statistics.mean(prom_in))
    print("SD")
    print(sd_in)
    print("SD average under number simulations")
    print(statistics.mean(sd_in))
    print("------------------------")
    print("Constraint satisfacion after self-optimization ({0}-{1})".format(set1+1,ss))
    print("Last value constraint satisfaction with Hebbian Learning")
    print(last_values_cs_fin)
    print("Average")
    print(prom_fin)
    print("Average under number simulations")
    print(statistics.mean(prom_fin))
    print("SD")
    print(sd_fin)
    print("SD average under number of simulations")
    print(statistics.mean(sd_fin))
    print("------------------------")
    print("Constraint satisfacion after self-optimization (stable) ({0}-{1})".format(ss+1,ss+set3))
    print("Average")
    print(prom_fin2)
    print("Average under number simulations")
    print(statistics.mean(prom_fin2))
    print("SD")
    print(sd_fin2)
    print("SD average under number of simulations")
    print(statistics.mean(sd_fin2))

    print("------------------------")
    n_arr = np.array(relaxs)
    n_arr_t = np.transpose(n_arr)
    resu = np.average(n_arr_t,axis=1)
    print("Average values local attractor (to be plotted)")
    print(*resu, sep=',\n')

    # Better comment for more than one simulation
    plt.plot([x for x in range(len(resu))], resu, 'b.')
    plt.axvline(x=set1,color='#BF4040')
    plt.axvline(x=ss,color='#BF4040')
    plt.show()

    timing=(time.time() - start_time)
    print("-----Total time -----",timing)
