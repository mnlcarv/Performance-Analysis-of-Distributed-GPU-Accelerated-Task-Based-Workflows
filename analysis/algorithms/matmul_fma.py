import dislib as ds
from dislib.data.array import Array
from pycompss.api.api import compss_barrier


# def run_matmul_fma(A, C, id_device, id_parameter, nr_algorithm_iteration):
def run_matmul_fma(experiment, dst_path_experiments):

    # Generate empty output matrix C to receive the result of the multiplication
    num_blocks = experiment.parameter.vl_grid_row_dimension
    elems_per_block = experiment.parameter.vl_block_row_dimension

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
    

    if experiment.parameter.ds_device == "GPU":

        # execution 1 - extract intra execution times with CUDA events
        compss_barrier()
        ds.dot(experiment.dataset, experiment.dataset, C, id_device=4, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)
        compss_barrier()

        # execution 2 - extract total and inter execution times with synchornized function calls
        compss_barrier()
        ds.dot(experiment.dataset, experiment.dataset, C, id_device=6, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)
        compss_barrier()

        # execution 3 - extract total execution time for GPU (id_device = 2)
        compss_barrier()
        ds.dot(experiment.dataset, experiment.dataset, C, id_device=experiment.parameter.id_device, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)
        compss_barrier()

    else:

        # execution 1 - extract intra execution times with synchornized function calls
        compss_barrier()
        ds.dot(experiment.dataset, experiment.dataset, C, id_device=3, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)
        compss_barrier()

        # execution 2 - extract total and inter execution times with synchornized function calls
        compss_barrier()
        ds.dot(experiment.dataset, experiment.dataset, C, id_device=5, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)
        compss_barrier()

        # execution 3 - extract total execution time for CPU (id_device = 1)
        compss_barrier()
        ds.dot(experiment.dataset, experiment.dataset, C, id_device=experiment.parameter.id_device, id_parameter=experiment.parameter.id_parameter, nr_algorithm_iteration=experiment.nr_algorithm_iteration)
        compss_barrier()