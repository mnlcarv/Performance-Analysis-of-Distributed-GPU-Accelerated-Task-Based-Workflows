# Introduction
This is the public repository of the paper "Performance Analysis of Distributed GPU-Accelerated Task-Based Workflows", which includes datasets, experimental results, and source code to extract different execution times from analytics algorithms.

## How to run
The experiments were executed with the distributed system [COMPSs](https://compss-doc.readthedocs.io/en/stable/index.html) using the version with bindings for Python (PyCOMPSs). The sample algorithms ( ```run_kmeans.py```, ```run_matmul.py```, and ```run_matmul_fma.py```) are executed with the ```runcompss``` command, as shown in the example below:
```
runcompss run_kmeans.py
```
After running the example, the raw execution time logs are saved to a csv file located at ```/experiments/results/tb_experiments_raw.csv```. Note that our scripts can be easily adapted to extract the execution times from other algorithms.

### Observations
1. In a cluster, the algorithms are executed via batch jobs. A job can be submitted using the ```enqueue_compss``` command. For more details, please check the basic queue commands in COMPSs [here](https://compss-doc.readthedocs.io/en/stable/Sections/03_Execution_Environments/03_Deployments/01_Master_worker/02_Supercomputers/03_Minotauro.html?highlight=supercomputer).
2. To extract data (de-)serialization times, it is necessary to activate tracing in COMPSs by including the extra argument ```--tracing=true``` to the execution command, as follows:
```
runcompss --tracing=true run_kmeans.py
```
The execution traces generated in COMPSs with [Extrae](https://tools.bsc.es/extrae) can be visualized using [Paraver](https://www.bsc.es/discover-bsc/organisation/scientific-structure/performance-tools/paraver). More information about traces in COMPSs and how to visualize them are available at the links below:

[COMPSs Applications Tracing](https://compss-doc.readthedocs.io/en/stable/Sections/05_Tools/03_Tracing/01_Apps_tracing.html)

[Trace Visualization in Paraver](https://compss-doc.readthedocs.io/en/stable/Sections/05_Tools/03_Tracing/02_Visualization.html?highlight=paraver)

## Dependencies
- COMPSs >= v3.0
- Python >= v3.7
- dislib >= v0.6.4
- NumPy >= v1.18.1
- CuPy >= v10

## Code references
[Matmul](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/dislib/data/array.py): dislib implementation of the Matmul algorithm.

[K-means](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/dislib/cluster/kmeans/base.py): dislib implementation of the K-means algorithm.

[Matmul FMA](https://compss-doc.readthedocs.io/en/stable/Sections/07_Sample_Applications/02_Python/04_Matmul.html?highlight=matmul): alternative algorithm, not belonging to the dislib library.

[dislib](https://github.com/bsc-wdc/dislib/tree/gpu-support): reference for dislib GPU support branch repository.


# Reproducibility

## Database
We stored parameter data in eight dimension tables (```DEVICE```, ```ALGORITHM```, ```FUNCTION```, ```CONFIGURATION```, ```RESOURCE```, ```DATASET```, ```PARAMETER_TYPE```, and ```PARAMETER```) and result data in a fact table (```EXPERIMENT_RAW```). We created a schema with these nine tables for each algorithm tested in our experiments. The database with all the metrics monitored in our experiments (except data deserialization and serialization times, which were obtained using [Paraver](https://www.bsc.es/discover-bsc/organisation/scientific-structure/performance-tools/paraver) and stored in CSV files) are available in the path [/reproducibility/database/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/database/).

Data deserialization and serialization times (columns 'Deserializing object' and 'Serializing object' in [Paraver's histogram](https://compss-doc.readthedocs.io/en/stable/Sections/05_Tools/03_Tracing/04_Analysis.html)) were stored in CSV files and they are available at [/reproducibility/paraver/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/paraver/).

## Queries
The queries used to extract the data plotted in each chart in our experiments are available at [/reproducibility/queries_charts/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/queries_charts/). To reproduce our results, load the tables (available in the path [/reproducibility/database/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/database/)) into a database and run our queries.

## Extension to other setups
We ran our experiments using the resources available at the [Minotauro](https://bsc.es/supportkc/docs/Minotauro/overview/) cluster. To extend our methodology to other setups, follow the steps below:

1. Make sure that the execution environment has the required dependencies. In our case, the distributed system COMPSs and additional libraries (dislib, Python, NumPy and CuPy) were needed to run the sample algorithms, but adapt this step according to the distributed system and libraries used.

2. Select an algorithm to study and identify the GPU-accelerated task(s) available on it. Make sure that the algorithm has two versions of the same task(s) (i.e. a version for CPUs and another for GPUs) to perform comparisons between CPUs and GPU.

3. Adapt the algorithm source code to extract the evaluated metrics presented (see Section 4.2 in the paper)*. Because parallel applications run asynchronously compared to sequential applications**, measuring the execution times in such applications requires synchronization barriers to ensure that all operations in a task are finished. For this reason, we prepared three execution modes for each processor type (CPU or GPU) controlled by the flag ```id_device``` as follows***:

- ```id_device = 1``` or ```id_device = 2```: execution with COMPSs synchronizations to extract total execution times (end-to-end execution) for CPUs and GPUs, respectively.

- ```id_device = 3``` or ```id_device = 4```: execution to extract task user code execution times for CPUs and GPUs (CUDA event synchronization is required for GPUs), respectively.

- ```id_device = 5 or id_device = 6```: execution with COMPSs synchronizations to extract parallel tasks execution times for CPUs and GPUs, respectively.

*Data deserialization and serialization times can be obtained with profiling tools (e.g. [Paraver](https://www.bsc.es/discover-bsc/organisation/scientific-structure/performance-tools/paraver)).

**More information about synchronization in COMPSs and CuPY are available in the links below:
[COMPSs synchronization](https://compss-doc.readthedocs.io/en/stable/Sections/02_App_Development/02_Python/01_2_Synchronization/01_API.html)
[CuPy synchronization](https://docs.cupy.dev/en/stable/user_guide/performance.html)

***Except for the execution mode ```id_device = 4```, which requires [CUDA events](https://docs.cupy.dev/en/stable/user_guide/performance.html), all the execution times are extracted using [Python's performance counter](https://docs.python.org/3/library/time.html).

4. Populate the dimension tables (```DEVICE```, ```ALGORITHM```, ```FUNCTION```, ```CONFIGURATION```, ```RESOURCE```, ```DATASET```, ```PARAMETER_TYPE```, and ```PARAMETER```) according to the available execution setup. Note that each row in the ```PARAMETER``` table must contain a unique set of execution parameters.

5. After executing the algorithm with one of the modes above, the raw execution time logs are saved to a csv file located at ```/experiments/results/tb_experiments_raw.csv```. Run the script [/reproducibility/scripts/python/insert_results.py](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/scripts/python/insert_results.py) to insert the results into the ```EXPERIMENT_RAW``` table using the column ```ID_PARAMETER``` to link the results to the each execution parameter in the ```PARAMETER``` table:
```
python /reproducibility/scripts/python/insert_results.py
```
We encourage users to use the same database schema design to store their results. The SQL ```CREATE``` scripts for the tables as well as their documentation are available at [/reproducibility/scripts/postgres/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/scripts/postgres/).


# Supplementary experiments
The experimental setup and results obtained in the supplementary experiments (i.e. additional algorithm (not belonging to the dislib library) and larger, skewed, and sparse datasets) are reported at [/experiments/results/supplementary_experiments/Report_Supplementary_Experiments.pdf](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/experiments/results/supplementary_experiments/Report_Supplementary_Experiments.pdf).