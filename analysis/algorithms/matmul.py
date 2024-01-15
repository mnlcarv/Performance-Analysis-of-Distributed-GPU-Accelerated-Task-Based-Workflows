import dislib as ds
from dislib.data.array import Array


def run_matmul(experiment, dst_path_experiments):

    transpose_a = transpose_b = experiment.parameter.bl_transpose_matrix
    
    if experiment.parameter.ds_device == "GPU":

        # execution 1 - extract intra execution times with CUDA events
        result = ds.matmul(experiment.dataset, experiment.dataset, transpose_a, transpose_b, id_device=4, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)

        # execution 2 - extract total and inter execution times with synchornized function calls
        result = ds.matmul(experiment.dataset, experiment.dataset, transpose_a, transpose_b, id_device=6, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)

        # execution 3 - extract total execution time for GPU (id_device = 2)
        result = ds.matmul(experiment.dataset, experiment.dataset, transpose_a, transpose_b, id_device=experiment.parameter.id_device, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)

    else:

        # execution 1 - extract intra execution times with synchornized function calls
        result = ds.matmul(experiment.dataset, experiment.dataset, transpose_a, transpose_b, id_device=3, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)

        # execution 2 - extract total and inter execution times with synchornized function calls
        result = ds.matmul(experiment.dataset, experiment.dataset, transpose_a, transpose_b, id_device=5, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)

        # execution 3 - extract total execution time for CPU (id_device = 1)
        result = ds.matmul(experiment.dataset, experiment.dataset, transpose_a, transpose_b, id_device=experiment.parameter.id_device, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)