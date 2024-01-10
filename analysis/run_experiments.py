import csv
import os
import time
import datetime
import pandas as pd
import algorithms.dislib as ds
from pycompss.api.api import compss_barrier
from algorithms.kmeans import run_kmeans
from algorithms.matmul import run_matmul
from algorithms.matmul_fma import run_matmul_fma
from dataset_generator import generate_dataset

def main():
    # Path of the "tb_parameters" table - CSV file
    src_path_parameters = "/parameters/tb_parameters.csv"
    # Path of the "tb_experiments_raw" table - CSV file
    dst_path_experiments = "/results/tb_experiments_raw.csv"

    # Reading "tb_parameters"
    param_file = os.path.join(src_path_parameters)
    df_parameters = pd.read_csv(param_file)

    # Defining the structure of "tb_experiments_raw" to log metrics
    header = ["id_parameter", "nr_algorithm_iteration", "nr_function_iteration", "nr_task", "start_total_execution_time", "end_total_execution_time", "start_inter_time_cpu", "end_inter_time_cpu", "intra_task_execution_full_func", "vl_intra_task_execution_time_device_func", "start_communication_time_1", "end_communication_time_1", "start_communication_time_2", "end_communication_time_2", "start_additional_time_1", "end_additional_time_1", "start_additional_time_2", "end_additional_time_2", "dt_processing"]
    # open "tb_experiments_raw" in the write mode
    f = open(dst_path_experiments, "w", encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(header)
    f.close()

    # Intilization of variable to store the current execution to measure the progress of the experiment
    current_execution = 0

    # Iterate over each row of "tb_parameters" (each row is a combination of execution parameters)
    for index, row in df_parameters.iterrows():

        current_execution += 1

        # Setting parameters variables
        id_parameter = row["id_parameter"]
        cd_parameter = row["cd_parameter"]
        cd_configuration = row["cd_configuration"]
        id_algorithm = row["id_algorithm"]
        ds_algorithm = row["ds_algorithm"]
        id_function = row["id_function"]
        ds_function = row["ds_function"]
        id_device = row["id_device"]
        ds_device = row["ds_device"]
        id_dataset = row["id_dataset"]
        id_resource = row["id_resource"]
        id_parameter_type = row["id_parameter_type"]
        ds_parameter_type = row["ds_parameter_type"]
        ds_parameter_attribute = row["ds_parameter_attribute"]
        ds_compss_version = row["ds_compss_version"]
        ds_dislib_version = row["ds_dislib_version"]
        ds_schdeuler = row["ds_schdeuler"]
        nr_cluster = row["nr_cluster"]
        bl_transpose_matrix = row["bl_transpose_matrix"]
        nr_iterations = row["nr_iterations"]
        vl_grid_row_dimension = row["vl_grid_row_dimension"]
        vl_grid_column_dimension = row["vl_grid_column_dimension"]
        vl_block_row_dimension = row["vl_block_row_dimension"]
        vl_block_column_dimension = row["vl_block_column_dimension"]
        vl_block_memory_size = row["vl_block_memory_size"]
        vl_block_memory_size_percent_cpu = row["vl_block_memory_size_percent_cpu"]
        vl_block_memory_size_percent_gpu = row["vl_block_memory_size_percent_gpu"]
        ds_resource = row["ds_resource"]
        nr_nodes = row["nr_nodes"]
        nr_computing_units_cpu = row["nr_computing_units_cpu"]
        nr_computing_units_gpu = row["nr_computing_units_gpu"]
        vl_memory_size_per_cpu_computing_unit = row["vl_memory_size_per_cpu_computing_unit"]
        vl_memory_size_per_gpu_computing_unit = row["vl_memory_size_per_gpu_computing_unit"]
        ds_dataset = row["ds_dataset"]
        vl_dataset_memory_size = row["vl_dataset_memory_size"]
        ds_data_type = row["ds_data_type"]
        vl_data_type_memory_size = row["vl_data_type_memory_size"]
        vl_dataset_dimension = row["vl_dataset_dimension"]
        vl_dataset_row_dimension = row["vl_dataset_row_dimension"]
        vl_dataset_column_dimension = row["vl_dataset_column_dimension"]
        nr_random_state = row["nr_random_state"]
        vl_data_sparsity = row["vl_data_sparsity"]
        vl_data_skewness = row["vl_data_skewness"]

        execution_progress = round((current_execution/df_parameters.shape[0])*100,2)
        print("\n@@@@@@ EXECUTION PROGRESS:",str(execution_progress),"%\n")
        print("nodes: ",str(nr_nodes),"\n")
        print("computing_units_cpu: ",str(nr_computing_units_cpu),"\n")
        print("vl_memory_size_per_cpu_computing_unit: ",str(vl_memory_size_per_cpu_computing_unit),"\n")
        print("computing_units_gpu: ",str(nr_computing_units_gpu),"\n")
        print("vl_memory_size_per_gpu_computing_unit: ",str(vl_memory_size_per_gpu_computing_unit),"\n")
        print("ds_device: ",str(ds_device),"\n")
        print("ds_parameter_type: ",str(ds_parameter_type),"\n")
        print("vl_dataset_memory_size: ",str(vl_dataset_memory_size),"\n")
        print("DATASET DIMENSION: vl_dataset_row_size x vl_dataset_column_size: ",str(vl_dataset_row_dimension)," x ",str(vl_dataset_column_dimension),"\n")
        print("GRID DIMENSION: vl_grid_row_size x vl_grid_column_size: ",str(vl_grid_row_dimension)," x ",str(vl_grid_column_dimension),"\n")
        print("BLOCK DIMENSION: vl_block_row_size x vl_block_column_size: ",str(vl_block_row_dimension)," x ",str(vl_block_column_dimension),"\n")
        print("vl_block_memory_size: ",str(vl_block_memory_size),"\n")


        # DATASET GENERATION
        # generate and load data into a ds-array
        dataset = generate_dataset(vl_dataset_row_dimension, vl_dataset_column_dimension, vl_block_row_dimension, vl_block_column_dimension, vl_grid_row_dimension, vl_grid_column_dimension, nr_random_state, vl_data_skewness, ds_algorithm)
        
        
        # Execute the experiment for N (nr_iterations) times with the same parameter set
        for i in range(nr_iterations + 1):
            
            iteration_experiment_time_start = datetime.datetime.now()

            print("\nEXPERIMENT ", id_parameter,"-------------- ITERATION ", i, " STARTED AT "+str(iteration_experiment_time_start)+"------------------\n")

            if ds_algorithm == "KMEANS":
                
                if ds_device == "GPU":

                    # execution 1 - extract intra execution times with CUDA events
                    run_kmeans(dataset=dataset,nr_cluster=nr_cluster,nr_random_state=nr_random_state,id_device=4,id_parameter=id_parameter,nr_algorithm_iteration=i)

                    # execution 2 - extract total and inter execution times with synchornized function calls
                    run_kmeans(dataset=dataset,nr_cluster=nr_cluster,nr_random_state=nr_random_state,id_device=6,id_parameter=id_parameter,nr_algorithm_iteration=i)

                else:

                    # execution 1 - extract intra execution times with synchornized function calls
                    run_kmeans(dataset=dataset,nr_cluster=nr_cluster,nr_random_state=nr_random_state,id_device=3,id_parameter=id_parameter,nr_algorithm_iteration=i)
                    
                    # execution 2 - extract total and inter execution times with synchornized function calls
                    run_kmeans(dataset=dataset,nr_cluster=nr_cluster,nr_random_state=nr_random_state,id_device=5,id_parameter=id_parameter,nr_algorithm_iteration=i)


                # execution 3 - extract total execution time for CPU (id_device = 1) and GPU (id_device = 2)
                compss_barrier()
                start_total_execution_time = time.perf_counter()

                run_kmeans(dataset=dataset,nr_cluster=nr_cluster,nr_random_state=nr_random_state,id_device=id_device,id_parameter=id_parameter,nr_algorithm_iteration=i)
                
                compss_barrier()
                end_total_execution_time = time.perf_counter()

                # log total execution time
                total_execution_time = end_total_execution_time - start_total_execution_time

                # open the log file in the append mode
                f = open(dst_path_experiments, "a", encoding='UTF8', newline='')

                # create a csv writer
                writer = csv.writer(f)

                # write the time data 
                var_null = 'NULL'
                data = [id_parameter, i, var_null, var_null, total_execution_time, var_null, var_null, var_null, var_null, var_null, var_null, var_null, datetime.datetime.now()]
                writer.writerow(data)
                f.close()


            elif ds_algorithm == "MATMUL_DISLIB":

                
                if ds_device == "GPU":

                    # execution 1 - extract intra execution times with CUDA events
                    run_matmul(x=dataset, bl_transpose_matrix=bl_transpose_matrix, id_device=4, id_parameter=id_parameter, nr_algorithm_iteration=i)

                    # execution 2 - extract total and inter execution times with synchornized function calls
                    run_matmul(x=dataset, bl_transpose_matrix=bl_transpose_matrix, id_device=6, id_parameter=id_parameter, nr_algorithm_iteration=i)

                    # execution 3 - extract total execution time for GPU (id_device = 2)
                    run_matmul(x=dataset, bl_transpose_matrix=bl_transpose_matrix, id_device=2, id_parameter=id_parameter, nr_algorithm_iteration=i)

                else:

                    # execution 1 - extract intra execution times with synchornized function calls
                    run_matmul(x=dataset, bl_transpose_matrix=bl_transpose_matrix, id_device=3, id_parameter=id_parameter, nr_algorithm_iteration=i)

                    # execution 2 - extract total and inter execution times with synchornized function calls
                    run_matmul(x=dataset, bl_transpose_matrix=bl_transpose_matrix, id_device=5, id_parameter=id_parameter, nr_algorithm_iteration=i)

                    # execution 3 - extract total execution time for CPU (id_device = 1)
                    run_matmul(x=dataset, bl_transpose_matrix=bl_transpose_matrix, id_device=1, id_parameter=id_parameter, nr_algorithm_iteration=i)

            elif ds_algorithm == "MATMUL_FMA":
                
                # Generate empty output matrix C to receive the result of the multiplication
                num_blocks = vl_grid_row_dimension
                elems_per_block = vl_block_row_dimension

                C = []
                for i in range(num_blocks):
                    for l in [C]:
                        l.append([])
                    # Keep track of blockId to initialize with different random seeds
                    bid = 0
                    for j in range(num_blocks):
                        C[-1].append(ds.generate_block(elems_per_block,
                                                    num_blocks,
                                                    set_to_zero=True))
                

                if ds_device == "GPU":

                    # execution 1 - extract intra execution times with CUDA events
                    compss_barrier()
                    run_matmul_fma(dataset, C, id_device=4, id_parameter=id_parameter, nr_algorithm_iteration=i)
                    compss_barrier()

                    # execution 2 - extract total and inter execution times with synchornized function calls
                    compss_barrier()
                    run_matmul_fma(dataset, C, id_device=6, id_parameter=id_parameter, nr_algorithm_iteration=i)
                    compss_barrier()

                    # execution 3 - extract total execution time for GPU (id_device = 2)
                    compss_barrier()
                    run_matmul_fma(dataset, C, id_device=2, id_parameter=id_parameter, nr_algorithm_iteration=i)
                    compss_barrier()

                else:

                    # execution 1 - extract intra execution times with synchornized function calls
                    compss_barrier()
                    run_matmul_fma(dataset, C, id_device=3, id_parameter=id_parameter, nr_algorithm_iteration=i)
                    compss_barrier()

                    # execution 2 - extract total and inter execution times with synchornized function calls
                    compss_barrier()
                    run_matmul_fma(dataset, C, id_device=5, id_parameter=id_parameter, nr_algorithm_iteration=i)
                    compss_barrier()

                    # execution 3 - extract total execution time for CPU (id_device = 1)
                    compss_barrier()
                    run_matmul_fma(dataset, C, id_device=1, id_parameter=id_parameter, nr_algorithm_iteration=i)
                    compss_barrier()

            else:
                print("Error: invalid algorithm!")
            
            iteration_experiment_time_end = datetime.datetime.now()
            iteration_experiment_time = (iteration_experiment_time_end - iteration_experiment_time_start).total_seconds()
            print("\nEXPERIMENT ", index+1,"-------------- ITERATION ", i, " FINISHED AT "+str(iteration_experiment_time_end)+" (TOTAL TIME: "+str(iteration_experiment_time)+") ------------------\n")

if __name__ == "__main__":
    main()