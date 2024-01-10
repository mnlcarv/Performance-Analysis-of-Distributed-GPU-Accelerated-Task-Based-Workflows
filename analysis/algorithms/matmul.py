import dislib as ds
from dislib.data.array import Array


def run_matmul(x, bl_transpose_matrix, id_device, id_parameter, nr_algorithm_iteration):

    transpose_a = transpose_b = bl_transpose_matrix
    result = ds.matmul(x, x, transpose_a, transpose_b, id_device=id_device, id_parameter=id_parameter, nr_algorithm_iteration=nr_algorithm_iteration)