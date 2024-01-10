# How to reproduce our results

Our results are stored in a relational database (Postgres). To reproduce our results, follow the steps below to create the tables, populate them with our collected data, and plot all charts of our experiments:

1. Insert your database connection details in the [database.ini](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/db_config/database.ini) file

2. Save the stored procedures [create_tables.sql](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/database/create_tables.sql) and [load_tables.sql](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/database/load_tables.sql) in your database

3. The queries used for each experiment are available in the folder [/database/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/database/). The queries and their description are described as follows:

| query name | desc |
| --- | --- |
| query_1.sql  | end-to-end performance analysis |
| query_2.sql  | task computational complexity in Matmul |
| query_5.sql  | varying clusters in K-means |
| query_6.sql  | varying data skew in Matmul |
| query_7.sql  | varying data skew in K-means |
| query_8.sql  | varying storage architecture and scheduling policy in Matmul |
| query_9.sql  | varying storage architecture and scheduling policy in K-means |
| query_10.sql  | correlation matrix of key features |
| query_11.sql  | analysis of task user code in Matmul FMA |
| query_12.sql  | varying data sparsity in Matmul |

4. Run [run_create_plots.py](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/run_create_plots.py) to plot all charts of our experiments (figures will be saved in the folder [/figures/](https://github.com/mnlcarv/Performance-Analysis-of-Distributed-GPU-Accelerated-Task-Based-Workflows/blob/main/reproducibility/figures/)): 
```
python run_create_plots.py
```