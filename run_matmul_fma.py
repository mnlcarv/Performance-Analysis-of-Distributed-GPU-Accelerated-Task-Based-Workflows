import time
import datetime
import csv
import dislib as ds
import numpy as np
from pycompss.api.api import compss_barrier

if __name__ == '__main__':

    # Path of the "tb_experiments_raw" table to store results in a CSV file
    dst_path_experiments = "experiments/results/tb_experiments_raw.csv"

    # Generate synthetic data set
    input_matrix_rows = 32000
    input_matrix_columns = 32000
    start_random_state = 170
    block_row_size = 8000
    block_column_size = 8000
    transpose_a = transpose_b = False
    
    num_blocks = int(input_matrix_rows/block_row_size) #grid_row_size
    elems_per_block = block_row_size

    A, C = [], []
    matrix_name = ["A"]
    for i in range(num_blocks):
        for l in [A, C]:
            l.append([])
        # Keep track of blockId to initialize with different random seeds
        bid = 0
        for j in range(num_blocks):
            for ix, l in enumerate([A]):
                l[-1].append(ds.generate_block(elems_per_block,
                                            num_blocks,
                                            random_state=start_random_state,
                                            bid=bid))
                bid += 1
            C[-1].append(ds.generate_block(elems_per_block,
                                        num_blocks,
                                        set_to_zero=True))

    # Execution modes:
    # id_device = 1: execution with synchornized function calls to extract total execution times for CPUs
    # id_device = 2: execution with synchornized function calls to extract total execution times for GPUs
    # id_device = 3: execution to extract task user code execution times for CPUs
    # id_device = 4: execution with CUDA events to extract task user code execution times for GPUs
    # id_device = 5: execution with synchornized function calls to extract parallel tasks execution times for CPUs
    # id_device = 6: execution with synchornized function calls to extract parallel tasks execution times for GPUs
    id_device = 1

    # Define number of iterations as 5
    nr_iterations = 5
    for i in range(nr_iterations):

        # Log only the total execution time (make sure that all tasks have finished by synchronizing them with compss_barrier)
        if id_device == 1 or id_device == 2:

            compss_barrier()
            start_total_execution_time = time.perf_counter()
            # Run Matmul FMA
            ds.dot(A, A, C, id_device=id_device, id_parameter=0, nr_algorithm_iteration=i)
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

            # Run Matmul FMA
            ds.dot(A, A, C, id_device=id_device, id_parameter=0, nr_algorithm_iteration=i)
