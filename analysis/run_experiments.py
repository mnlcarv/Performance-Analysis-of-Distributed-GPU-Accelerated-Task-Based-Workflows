import csv
import os
import datetime
import pandas as pd
from algorithms.kmeans import run_kmeans
from algorithms.matmul import run_matmul
from algorithms.matmul_fma import run_matmul_fma
from dataset_generator import generate_dataset

def main():
    # Path of the "tb_parameters" table - CSV file
    src_path_parameters = "parameters/tb_parameters.csv"
    # Path of the "tb_experiments_raw" table - CSV file
    dst_path_experiments = "results/tb_experiments_raw.csv"

    # Reading "tb_parameters"
    param_file = os.path.join(src_path_parameters)
    df_parameters = pd.read_csv(param_file)

    # Defining the structure of "tb_experiments_raw" to log metrics
    header = read_column_names_csv(dst_path_experiments)[0]

    # open "tb_experiments_raw" in the write mode to clean previous results
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
        parameter = Parameter(*row)

        execution_progress = round((current_execution/df_parameters.shape[0])*100,2)
        print("\n@@@@@@ EXECUTION PROGRESS:",str(execution_progress),"%\n")
        print("algorithm: ",str(parameter.ds_algorithm),"\n")
        print("device: ",str(parameter.ds_device),"\n")
        print("dataset memomory size: ",str(parameter.vl_dataset_memory_size),"\n")
        print("dataset dimension: ",str(parameter.vl_dataset_row_dimension)," x ",str(parameter.vl_dataset_column_dimension),"\n")
        print("grid dimension: ",str(parameter.vl_grid_row_dimension)," x ",str(parameter.vl_grid_column_dimension),"\n")
        print("block dimension: ",str(parameter.vl_block_row_dimension)," x ",str(parameter.vl_block_column_dimension),"\n")
        print("block memory size: ",str(parameter.vl_block_memory_size),"\n")


        # DATASET GENERATION
        dataset = generate_dataset(parameter.vl_dataset_row_dimension, parameter.vl_dataset_column_dimension, parameter.vl_block_row_dimension, parameter.vl_block_column_dimension, parameter.vl_grid_row_dimension, parameter.vl_grid_column_dimension, parameter.nr_random_state, parameter.vl_data_skewness, parameter.ds_algorithm)
        
        
        # Execute the experiment for N (nr_iterations) times with the same parameter set
        for i in range(parameter.nr_iterations + 1):
            
            iteration_experiment_time_start = datetime.datetime.now()

            print("\nEXPERIMENT ", parameter.id_parameter,"-------------- ITERATION ", i, " STARTED AT "+str(iteration_experiment_time_start)+"------------------\n")

            # EXPERIMENT EXECUTION
            experiment = Experiment(parameter, dataset, nr_algorithm_iteration = i)
            experiment.run(dst_path_experiments)
            
            iteration_experiment_time_end = datetime.datetime.now()
            iteration_experiment_time = (iteration_experiment_time_end - iteration_experiment_time_start).total_seconds()
            print("\nEXPERIMENT ", index+1,"-------------- ITERATION ", i, " FINISHED AT "+str(iteration_experiment_time_end)+" (TOTAL TIME: "+str(iteration_experiment_time)+") ------------------\n")


def read_column_names_csv(dst_path_experiments):
    with open(dst_path_experiments) as csv_file:
 
        # creating an object of csv reader with the delimiter as ,
        csv_reader = csv.reader(csv_file, delimiter = ',')
    
        # list to store the names of columns
        list_of_column_names = []
    
        # loop to iterate through the rows of csv
        for row in csv_reader:
    
            # adding the first row
            list_of_column_names.append(row)
    
            # breaking the loop after the first iteration itself
            break
    return list_of_column_names


# Class Parameter 
class Parameter:
    def __init__(self, id_parameter, cd_parameter, cd_configuration, id_algorithm, ds_algorithm,
                 id_function, ds_function, id_device, ds_device, id_dataset, id_resource,
                 id_parameter_type, ds_parameter_type, ds_parameter_attribute, ds_compss_version,
                 ds_dislib_version, ds_schdeuler, nr_cluster, bl_transpose_matrix, nr_iterations,
                 vl_grid_row_dimension, vl_grid_column_dimension, vl_block_row_dimension,
                 vl_block_column_dimension, vl_block_memory_size, vl_block_memory_size_percent_cpu,
                 vl_block_memory_size_percent_gpu, ds_resource, nr_nodes, nr_computing_units_cpu,
                 nr_computing_units_gpu, vl_memory_size_per_cpu_computing_unit, vl_memory_size_per_gpu_computing_unit,
                 ds_dataset, vl_dataset_memory_size, ds_data_type, vl_data_type_memory_size, vl_dataset_dimension,
                 vl_dataset_row_dimension, vl_dataset_column_dimension, nr_random_state, vl_data_sparsity, vl_data_skewness):
        
        self.id_parameter = id_parameter
        self.cd_parameter = cd_parameter
        self.cd_configuration = cd_configuration
        self.id_algorithm = id_algorithm
        self.ds_algorithm = ds_algorithm
        self.id_function = id_function
        self.ds_function = ds_function
        self.id_device = id_device
        self.ds_device = ds_device
        self.id_dataset = id_dataset
        self.id_resource = id_resource
        self.id_parameter_type = id_parameter_type
        self.ds_parameter_type = ds_parameter_type
        self.ds_parameter_attribute = ds_parameter_attribute
        self.ds_compss_version = ds_compss_version
        self.ds_dislib_version = ds_dislib_version
        self.ds_schdeuler = ds_schdeuler
        self.nr_cluster = nr_cluster
        self.bl_transpose_matrix = bl_transpose_matrix
        self.nr_iterations = nr_iterations
        self.vl_grid_row_dimension = vl_grid_row_dimension
        self.vl_grid_column_dimension = vl_grid_column_dimension
        self.vl_block_row_dimension = vl_block_row_dimension
        self.vl_block_column_dimension = vl_block_column_dimension
        self.vl_block_memory_size = vl_block_memory_size
        self.vl_block_memory_size_percent_cpu = vl_block_memory_size_percent_cpu
        self.vl_block_memory_size_percent_gpu = vl_block_memory_size_percent_gpu
        self.ds_resource = ds_resource
        self.nr_nodes = nr_nodes
        self.nr_computing_units_cpu = nr_computing_units_cpu
        self.nr_computing_units_gpu = nr_computing_units_gpu
        self.vl_memory_size_per_cpu_computing_unit = vl_memory_size_per_cpu_computing_unit
        self.vl_memory_size_per_gpu_computing_unit = vl_memory_size_per_gpu_computing_unit
        self.ds_dataset = ds_dataset
        self.vl_dataset_memory_size = vl_dataset_memory_size
        self.ds_data_type = ds_data_type
        self.vl_data_type_memory_size = vl_data_type_memory_size
        self.vl_dataset_dimension = vl_dataset_dimension
        self.vl_dataset_row_dimension = vl_dataset_row_dimension
        self.vl_dataset_column_dimension = vl_dataset_column_dimension
        self.nr_random_state = nr_random_state
        self.vl_data_sparsity = vl_data_sparsity
        self.vl_data_skewness = vl_data_skewness

# Class experiment
class Experiment:
    def __init__(self, parameter, dataset, nr_algorithm_iteration):

        self.parameter = parameter
        self.dataset = dataset
        self.nr_algorithm_iteration = nr_algorithm_iteration

    def run(self, dst_path_experiments):
        
        if self.parameter.ds_algorithm == 'KMEANS':
            run_kmeans(self, dst_path_experiments)
        elif self.parameter.ds_algorithm == 'MATMUL_DISLIB':
            run_matmul(self, dst_path_experiments)
        elif self.parameter.ds_algorithm == 'MATMUL_FMA':
            run_matmul_fma(self, dst_path_experiments)
        else:
            print('\nAlgorithm not supported!')


if __name__ == "__main__":
    main()