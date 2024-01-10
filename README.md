# Introduction
This is the public repository of an empirical study about the execution performance of task-based workflows on a High Performance Computing (HPC) infrastructure composed of heterogeneous CPU-GPU clusters. The analysis method, results, and artifacts used in the study are detailed below.


# Analysis method
Our focus of this study is on analyzing the impact of GPUs on task-based workflows commonly used data science pipelines, which are composed of multiple processing stages that perform different tasks to move data from one stage to a next
stage (typically represented by a Directed Acyclic Graph (DAG)).

## Task processing stages
Generally, task processing involves data computation, represented by the task user code, and data movement, represented mainly by data (de-)serialization. Task user code has the following processing stages:

- Parallel fraction
- Serial fraction
- CPU-GPU communication

Consider for example the task *partial_sum* task in the [K-means](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/analysis/algorithms/dislib/cluster/kmeans/base.py) algorithm ([dislib](https://github.com/bsc-wdc/dislib/tree/gpu-support) implementation). The processing stages of this task are highlighted in the figure below:

![alt text](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/figures/sample_code.png?raw=true)

## Monitored metrics
We consider metrics related to task execution according to its processing stages:

- **Task user code metrics**
    - **Serial fraction execution time**: time to process the serial fraction of the user code
    - **Parallel fraction execution time**: time to process the parallel fraction of the user code
    - **CPU-GPU communication time**: time to move data from CPU to GPU or vice versa
    - **User code execution time**: time per task user code; i.e., summary of serial fraction, parallel fraction, and CPU-GPU communication times

- **Data movement overheads**
    - **Deserialization time**: time to read data from storage (e.g., disk) and load it to main memory
    - **Serialization time**: time to write data from main memory to storage

- **Task level metric**
    - **Parallel task execution time**: time to run in parallel tasks placed in the same level in the DAG, considering all overheads related to data movement

## Database of experiments
We stored the results of our experiments in a database using a star-schema model with dimension tables (for execution parameters) and a fact table (for the monitored metrics). We used a relational database (Postgres) to store the execution parameters and results with all the metrics monitored in our experiments (except data deserialization and serialization times, which were obtained using [Paraver](https://www.bsc.es/discover-bsc/organisation/scientific-structure/performance-tools/paraver) and stored in separated csv files). We stored the parameters data in eight dimension tables (```DEVICE```, ```ALGORITHM```, ```FUNCTION```, ```CONFIGURATION```, ```RESOURCE```, ```DATASET```, ```PARAMETER_TYPE```, and ```PARAMETER```) and the result data in a fact table (```EXPERIMENT_RAW```). Each row in the ```PARAMETER``` table represents a unique combination of execution parameters, which are identified in the ```EXPERIMENT_RAW``` table by the column ```ID_PARAMETER```. A detailed documentation of the tables is available in [DOCUMENTATION_TABLES.csv](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/database/documentation_tables/DOCUMENTATION_TABLES.csv).

## How to run experiments
We ran our experiments using [COMPSs](https://compss-doc.readthedocs.io/en/stable/index.html) and algorithms from the [dislib](https://github.com/bsc-wdc/dislib/tree/gpu-support) library. Given a set of execution parameters related to the task algorithm, dataset, resources, and system employed, we execute the algorithms and extract the monitored metrics in an intermediate log file that is further inserted in our fact table (```EXPERIMENT_RAW```).


# Repo structure
Our repository is organized as into two main subfolders: [/reproducibility/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/) and [/analysis/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/analysis/).

### Reproducibility
The database of with the raw data, queries and scripts used to plot the charts of all our experiments is available in [/reproducibility/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/).

### Analysis
Instructions about how to run experiments on a compatible system (i.e., COMPSs in a distributed infrastructure), as well as a methodology to extend our analysis method to other setups are available in [/analysis/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/analysis/).  