import time
import datetime
import csv
import dislib as ds
import numpy as np
from dislib.cluster import KMeans
from pycompss.api.api import compss_barrier

if __name__ == '__main__':

    # Path of the "tb_experiments_raw" table to store results in a CSV file
    dst_path_experiments = "experiments/results/tb_experiments_raw.csv"

    # Generate synthetic data set
    input_matrix_rows = 12500000
    input_matrix_columns = 10
    start_random_state = 170
    block_row_size = 3125000
    block_column_size = 10
    n_clusters = 1000
    x_np = np.random.random((input_matrix_rows, input_matrix_columns))
    x = ds.array(x_np, block_size=(block_row_size, block_column_size))

    # Execution modes:
    # id_device = 1: execution with synchornized function calls to extract total execution times for CPUs
    # id_device = 2: execution with synchornized function calls to extract total execution times for GPUs
    # id_device = 3: execution with synchornized function calls to extract task user code execution times for CPUs
    # id_device = 4: execution with synchornized function calls and CUDA events to extract task user code execution times for GPUs
    # id_device = 5: execution with synchornized function calls to extract parallel tasks execution times for CPUs
    # id_device = 6: execution with synchornized function calls to extract parallel tasks execution times for GPUs
    id_device = 2
    
    # Define number of iterations as 5
    nr_iterations = 0
    for i in range(nr_iterations + 1):

        # Log only the total execution time (make sure that all tasks have finished by synchronizing them with compss_barrier)
        if id_device == 1 or id_device == 2:

            compss_barrier()
            start_total_execution_time = time.perf_counter()
            # Run KMeans using dislib
            kmeans = KMeans(n_clusters=n_clusters, random_state=start_random_state, id_device=id_device, max_iter=5, tol=0, arity=48)
            kmeans.fit(x)
            compss_barrier()
            end_total_execution_time = time.perf_counter()

            # Open the log file in the append mode
            f = open(dst_path_experiments, "a", encoding='UTF8', newline='')

            # Create a csv writer
            writer = csv.writer(f)

            # Write data and close the log file
            data = [0, i, 'NULL', 'NULL', start_total_execution_time, end_total_execution_time, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', datetime.datetime.now()]
            writer.writerow(data)
            f.close()
        
        # Log intermediate task processing stages according to id_device selected
        else:

            # Run KMeans using dislib
            kmeans = KMeans(n_clusters=n_clusters, random_state=start_random_state, id_device=id_device, max_iter=5, tol=0, arity=48)
            kmeans.fit(x)
