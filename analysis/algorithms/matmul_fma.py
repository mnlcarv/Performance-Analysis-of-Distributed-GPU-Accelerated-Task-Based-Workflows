import dislib as ds
from dislib.data.array import Array


def run_matmul_fma(A, C, id_device, id_parameter, nr_algorithm_iteration):

    ds.dot(A=A, A=A, C=C, id_device=id_device, id_parameter=id_parameter, nr_algorithm_iteration=nr_algorithm_iteration)