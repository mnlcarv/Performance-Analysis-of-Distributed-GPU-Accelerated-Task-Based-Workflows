# Introduction
This is the public repository of paper "Performance Analysis of Distributed GPU-Accelerated Task-Based Workflows", which includes the source code to extract different execution times from dislib implementations of Matmul and K-means.

## How to Run
The demo applications (```run_matmul.py``` and ```run_kmeans.py```) are executed with the "runcompss" command as shown in the example below:
```
runcompss run_matmul.py
```
In a cluster, the applications are submitted via batch jobs. A job can be submitted using "enqueue_compss" command. For more details, please check the official documentation of COMPSs [here](https://compss-doc.readthedocs.io/en/stable/index.html).

After running the example, the raw execution time logs are saved to a csv file located at ```experiments/results/tb_experiments_raw.csv```.

## Dependencies
- pycompss
- dislib
- numpy
- cupy

## Code References
[Matmul](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/dislib/data/array.py): dislib implementation of the Matmul algorithm.

[K-means](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/dislib/cluster/kmeans/base.py): dislib implementation of the K-means algorithm.

[dislib](https://github.com/bsc-wdc/dislib/tree/gpu-support): reference for dislib GPU support branch repository.
