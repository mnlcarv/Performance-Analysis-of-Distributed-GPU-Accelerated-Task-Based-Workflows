import dislib as ds
from dislib.cluster import KMeans
from dislib.data.array import Array


def run_kmeans(dataset,nr_cluster,nr_random_state,id_device,id_parameter,nr_algorithm_iteration):

    kmeans = KMeans(n_clusters=nr_cluster, random_state=nr_random_state, id_device=id_device, id_parameter=id_parameter, nr_algorithm_iteration=nr_algorithm_iteration, max_iter=5, tol=0, arity=48)
    kmeans.fit(dataset)