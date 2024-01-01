# Introduction
This is the public repository of the paper "Performance Analysis of Distributed GPU-Accelerated Task-Based Workflows", which includes the source code to extract different execution times from the dislib implementations of Matmul and K-means, as well as the COMPSs implementation of Matmul FMA.

## How to run
The demo applications (```run_matmul.py``` and ```run_kmeans.py```) are executed with the ```runcompss``` command, as shown in the example below:
```
runcompss run_matmul.py
```
After running the example, the raw execution time logs are saved to a csv file located at ```/experiments/results/tb_experiments_raw.csv```.

### Observations
1. In a cluster, the applications are submitted via batch jobs. A job can be submitted using the ```enqueue_compss``` command. For more details, please check the official documentation of COMPSs [here](https://compss-doc.readthedocs.io/en/stable/index.html).
2. To extract data (de-)serialization times, it is necessary to activate tracing in COMPSs by including the extra argument ```--tracing=true``` to the execution command, as follows:
```
runcompss --tracing=true run_matmul.py
```
The execution traces generated in COMPSs with [Extrae](https://tools.bsc.es/extrae) can be visualized using [Paraver](https://www.bsc.es/discover-bsc/organisation/scientific-structure/performance-tools/paraver). More information about traces in COMPSs and how to visualize them are available at the links below:
[COMPSs Applications Tracing](https://compss-doc.readthedocs.io/en/stable/Sections/05_Tools/03_Tracing/01_Apps_tracing.html)
[Trace Visualization in Paraver](https://compss-doc.readthedocs.io/en/stable/Sections/05_Tools/03_Tracing/02_Visualization.html?highlight=paraver)

## Dependencies
- pycompss
- dislib
- numpy
- cupy

## Code references
[Matmul](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/dislib/data/array.py): dislib implementation of the Matmul algorithm.

[K-means](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/dislib/cluster/kmeans/base.py): dislib implementation of the K-means algorithm.

[Matmul FMA](https://compss-doc.readthedocs.io/en/stable/Sections/07_Sample_Applications/02_Python/04_Matmul.html?highlight=matmul): alternative algorithm, not belonging to the dislib library.

[dislib](https://github.com/bsc-wdc/dislib/tree/gpu-support): reference for dislib GPU support branch repository.


# Data resources
We stored parameter data in eight dimension tables (DEVICE, ALGORITHM, FUNCTION, CONFIGURATION, RESOURCE, DATASET, PARAMETER_TYPE, and PARAMETER) and result data in a fact table (EXPERIMENT_RAW). We created a schema with these nine tables for each algorithm tested. The data resources are available in the path [/data/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/data/). 

Data (de-)serialization times (columns 'Deserializing object' and 'Serializing object' in [Paraver's histogram](https://compss-doc.readthedocs.io/en/stable/Sections/05_Tools/03_Tracing/04_Analysis.html)) were stored in CSV files and they are available at [/data/paraver/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/data/paraver/): 

## Extension to other setups
We ran our experiments using the resources available at the [Minotauro](https://bsc.es/supportkc/docs/Minotauro/overview/) cluster. To extend our methodology to other setups, we encourage users to use the same database schema design to store their data (i.e. resources, algorithms, datasets, etc). The SQL ```CREATE``` scripts for the tables as well as their data dictionary are available at [/experiments/scripts/create_tables/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/experiments/scripts/create_tables/).


# Reproducibility
The queries used to extract the data plotted in the charts are available at [/experiments/scripts/data_charts/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/experiments/scripts/data_charts/).


# Supplementary experiments
The results obtained in the supplementary experiments (i.e. additional algorithm (not belonging to dislib library) and larger, skewed, and sparse datasets) are available at [/experiments/results/supplementary_experiments/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/experiments/results/supplementary_experiments/).