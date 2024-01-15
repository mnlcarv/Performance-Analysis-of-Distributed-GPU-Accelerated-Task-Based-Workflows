import csv
import os
import time
import datetime
import dislib as ds
from dislib.cluster import KMeans
from pycompss.api.api import compss_barrier

def run_kmeans(experiment, dst_path_experiments):

    if experiment.parameter.ds_device == "GPU":

        # execution 1 - extract intra execution times with CUDA events
        kmeans = KMeans(n_clusters=experiment.parameter.nr_cluster, random_state=experiment.parameter.nr_random_state, id_device=4, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration, max_iter=5, tol=0, arity=48)
        kmeans.fit(experiment.dataset)

        # execution 2 - extract total and inter execution times with synchornized function calls
        kmeans = KMeans(n_clusters=experiment.parameter.nr_cluster, random_state=experiment.parameter.nr_random_state, id_device=6, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration, max_iter=5, tol=0, arity=48)
        kmeans.fit(experiment.dataset)

    else:

        # execution 1 - extract intra execution times with synchornized function calls
        kmeans = KMeans(n_clusters=experiment.parameter.nr_cluster, random_state=experiment.parameter.nr_random_state, id_device=3, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration, max_iter=5, tol=0, arity=48)
        kmeans.fit(experiment.dataset)
        
        # execution 2 - extract total and inter execution times with synchornized function calls
        kmeans = KMeans(n_clusters=experiment.parameter.nr_cluster, random_state=experiment.parameter.nr_random_state, id_device=5, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration, max_iter=5, tol=0, arity=48)
        kmeans.fit(experiment.dataset)

    # execution 3 - extract total execution time for CPU (id_device = 1) and GPU (id_device = 2)
    compss_barrier()
    start_total_execution_time = time.perf_counter()

    kmeans = KMeans(n_clusters=experiment.parameter.nr_cluster, random_state=experiment.parameter.nr_random_state, id_device=experiment.parameter.id_device, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration, max_iter=5, tol=0, arity=48)
    kmeans.fit(experiment.dataset)
    
    compss_barrier()
    end_total_execution_time = time.perf_counter()

    # open the log file in the append mode
    f = open(dst_path_experiments, "a", encoding='UTF8', newline='')

    # create a csv writer
    writer = csv.writer(f)

    # write the time data 
    var_null = 'NULL'
    data = [experiment.parameter.id_parameter, experiment.nr_algorithm_iteration, var_null, var_null, start_total_execution_time, end_total_execution_time, var_null, var_null, var_null, var_null, var_null, var_null, var_null, var_null, var_null, var_null, var_null, var_null, datetime.datetime.now()]
    writer.writerow(data)
    f.close()