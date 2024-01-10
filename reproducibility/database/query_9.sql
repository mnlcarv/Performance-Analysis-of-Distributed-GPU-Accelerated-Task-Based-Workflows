                        -- EXPERIMENT RAW QUERY
                        SELECT
                        Y.VL_TOTAL_EXECUTION_TIME,
                        Y.VL_INTER_TASK_EXECUTION_TIME,
                        (Y.VL_INTER_TASK_EXECUTION_TIME - Y.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC) AS VL_INTER_TASK_OVERHEAD_TIME,
                        Y.VL_INTER_TASK_EXECUTION_TIME - (Y.VL_INTER_TASK_EXECUTION_TIME - Y.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC) AS VL_INTER_TASK_EXECUTION_TIME_FREE_OVERHEAD,
                        Y.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
                        Y.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
                        Y.VL_COMMUNICATION_TIME,
                        Y.VL_ADDITIONAL_TIME,
                        (Y.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC + Y.VL_COMMUNICATION_TIME) AS VL_INTRA_TASK_EXECUTION_TIME_FREE_ADDITIONAL,
						Y.VL_STD_TOTAL_EXECUTION_TIME,
						Y.VL_STD_INTER_TASK_EXECUTION_TIME,
						Y.VL_STD_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
						Y.VL_STD_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
						Y.VL_STD_COMMUNICATION_TIME,
                        ROUND(((Y.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC)/(Y.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC-Y.VL_COMMUNICATION_TIME))::numeric,2) AS P_FRACTION,
						Y.ID_PARAMETER,
                        Y.CD_PARAMETER,
                        Y.CD_CONFIGURATION,
                        Y.ID_ALGORITHM,
                        Y.DS_ALGORITHM,
                        Y.ID_FUNCTION,
                        Y.DS_FUNCTION,
                        Y.ID_DEVICE,
                        Y.DS_DEVICE,
                        Y.ID_DATASET,
                        Y.ID_RESOURCE,
                        Y.ID_PARAMETER_TYPE,
                        Y.DS_PARAMETER_TYPE,
                        Y.DS_PARAMETER_ATTRIBUTE,
                        Y.NR_ITERATIONS,
                        Y.VL_GRID_ROW_DIMENSION,
                        Y.VL_GRID_COLUMN_DIMENSION,
                        Y.VL_GRID_ROW_X_COLUMN_DIMENSION,
						Y.VL_CONCAT_GRID_ROW_X_COLUMN_DIMENSION_BLOCK_SIZE_MB,
                        Y.VL_CONCAT_DATASET_MB_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
						Y.VL_CONCAT_BLOCK_SIZE_MB_NR_TASKS,
                        Y.VL_CONCAT_BLOCK_SIZE_MB_GRID_ROW_X_COLUMN_DIMENSION,
                        Y.VL_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                        Y.VL_CONCAT_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                        Y.VL_BLOCK_ROW_DIMENSION,
                        Y.VL_BLOCK_COLUMN_DIMENSION,
                        Y.VL_BLOCK_ROW_X_COLUMN_DIMENSION,
						ROUND(Y.VL_BLOCK_MEMORY_SIZE*1e-6,2) AS VL_BLOCK_MEMORY_SIZE,
                        Y.VL_BLOCK_MEMORY_SIZE_PERCENT_CPU,
                        Y.VL_BLOCK_MEMORY_SIZE_PERCENT_GPU,
                        Y.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                        Y.VL_CONCAT_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                        Y.DS_RESOURCE,
                        Y.NR_NODES,
                        Y.NR_COMPUTING_UNITS_CPU,
                        (Y.NR_NODES-1)*Y.NR_COMPUTING_UNITS_CPU AS NR_TOTAL_COMPUTING_UNITS_CPU,
                        (Y.NR_NODES-1) || ' (' || (Y.NR_NODES-1)*Y.NR_COMPUTING_UNITS_CPU || ')' AS NR_CONCAT_NODES_TOTAL_COMPUTING_UNITS_CPU,
                        Y.NR_COMPUTING_UNITS_GPU,
                        Y.VL_MEMORY_SIZE_PER_CPU_COMPUTING_UNIT,
                        Y.VL_MEMORY_SIZE_PER_GPU_COMPUTING_UNIT,
                        Y.DS_DATASET,
                        Y.VL_DATASET_MEMORY_SIZE,
                        Y.DS_DATA_TYPE,
                        Y.VL_DATA_TYPE_MEMORY_SIZE,
                        Y.VL_DATASET_DIMENSION,
                        Y.VL_DATASET_ROW_DIMENSION,
                        Y.VL_DATASET_COLUMN_DIMENSION,
                        Y.VL_DATASET_ROW_X_COLUMN_DIMENSION,
                        Y.NR_RANDOM_STATE,
                        CASE
                            WHEN Y.DS_DEVICE = 'CPU' AND Y.VL_DATA_SKEWNESS = 0 THEN 'CPU NOT SKEWED'
                            WHEN Y.DS_DEVICE = 'CPU' AND Y.VL_DATA_SKEWNESS = 0.5 THEN 'CPU SKEWED'
                            WHEN Y.DS_DEVICE = 'GPU' AND Y.VL_DATA_SKEWNESS = 0 THEN 'GPU NOT SKEWED'
                            WHEN Y.DS_DEVICE = 'GPU' AND Y.VL_DATA_SKEWNESS = 0.5 THEN 'GPU SKEWED'
                            ELSE ''
                        END AS DEVICE_SKEWNESS
                        FROM
                        (
                            SELECT
                            AVG(X.VL_TOTAL_EXECUTION_TIME) AS VL_TOTAL_EXECUTION_TIME,
                            CASE
								WHEN X.DS_FUNCTION = 'MATMUL_FUNC' THEN AVG(X.VL_INTER_TASK_EXECUTION_TIME)
								WHEN X.DS_FUNCTION = 'ADD_FUNC' THEN AVG(X.VL_INTER_TASK_EXECUTION_TIME)*ceil(ln(X.VL_GRID_ROW_DIMENSION)/ln(2))
								ELSE AVG(X.VL_INTER_TASK_EXECUTION_TIME)
							END AS VL_INTER_TASK_EXECUTION_TIME,
                            AVG(X.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC) AS VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
                            AVG(X.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC) AS VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
                            AVG(X.VL_COMMUNICATION_TIME_1) AS VL_COMMUNICATION_TIME_1,
                            AVG(X.VL_COMMUNICATION_TIME_2) AS VL_COMMUNICATION_TIME_2,
                            AVG(X.VL_COMMUNICATION_TIME) AS VL_COMMUNICATION_TIME,
                            AVG(X.VL_ADDITIONAL_TIME_1) AS VL_ADDITIONAL_TIME_1,
                            AVG(X.VL_ADDITIONAL_TIME_2) AS VL_ADDITIONAL_TIME_2,
                            AVG(X.VL_ADDITIONAL_TIME) AS VL_ADDITIONAL_TIME,
                            STDDEV(X.VL_TOTAL_EXECUTION_TIME) AS VL_STD_TOTAL_EXECUTION_TIME,
                            STDDEV(X.VL_INTER_TASK_EXECUTION_TIME) AS VL_STD_INTER_TASK_EXECUTION_TIME,
                            STDDEV(X.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC) AS VL_STD_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
                            STDDEV(X.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC) AS VL_STD_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
                            STDDEV(X.VL_COMMUNICATION_TIME) AS VL_STD_COMMUNICATION_TIME,
                            X.ID_PARAMETER,
                            X.CD_PARAMETER,
                            X.CD_CONFIGURATION,
                            X.ID_ALGORITHM,
                            X.DS_ALGORITHM,
                            X.ID_FUNCTION,
                            X.DS_FUNCTION,
                            X.ID_DEVICE,
                            X.DS_DEVICE,
                            X.ID_DATASET,
                            X.ID_RESOURCE,
                            X.ID_PARAMETER_TYPE,
                            X.DS_PARAMETER_TYPE,
                            X.DS_PARAMETER_ATTRIBUTE,
                            X.NR_ITERATIONS,
                            X.VL_GRID_ROW_DIMENSION,
                            X.VL_GRID_COLUMN_DIMENSION,
                            X.VL_GRID_ROW_X_COLUMN_DIMENSION,
							X.VL_CONCAT_GRID_ROW_X_COLUMN_DIMENSION_BLOCK_SIZE_MB,
                            X.VL_CONCAT_DATASET_MB_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
							X.VL_CONCAT_BLOCK_SIZE_MB_NR_TASKS,
                            X.VL_CONCAT_BLOCK_SIZE_MB_GRID_ROW_X_COLUMN_DIMENSION,
                            X.VL_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                            X.VL_CONCAT_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                            X.VL_BLOCK_ROW_DIMENSION,
                            X.VL_BLOCK_COLUMN_DIMENSION,
                            X.VL_BLOCK_ROW_X_COLUMN_DIMENSION,
							ROUND(X.VL_BLOCK_MEMORY_SIZE*1e-6,2) AS VL_BLOCK_MEMORY_SIZE,
                            X.VL_BLOCK_MEMORY_SIZE_PERCENT_CPU,
                            X.VL_BLOCK_MEMORY_SIZE_PERCENT_GPU,
                            X.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                            X.VL_CONCAT_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                            X.DS_RESOURCE,
                            X.NR_NODES,
                            X.NR_COMPUTING_UNITS_CPU,
                            X.NR_COMPUTING_UNITS_GPU,
                            X.VL_MEMORY_SIZE_PER_CPU_COMPUTING_UNIT,
                            X.VL_MEMORY_SIZE_PER_GPU_COMPUTING_UNIT,
                            X.DS_DATASET,
                            X.VL_DATASET_MEMORY_SIZE,
                            X.DS_DATA_TYPE,
                            X.VL_DATA_TYPE_MEMORY_SIZE,
                            X.VL_DATASET_DIMENSION,
                            X.VL_DATASET_ROW_DIMENSION,
                            X.VL_DATASET_COLUMN_DIMENSION,
                            X.VL_DATASET_ROW_X_COLUMN_DIMENSION,
                            X.NR_RANDOM_STATE,
							X.VL_DATA_SPARSITY,
							X.VL_DATA_SKEWNESS
                            FROM
                            (
                                SELECT
                                    A.ID_EXPERIMENT,
                                    A.VL_TOTAL_EXECUTION_TIME,
                                    A.VL_INTER_TASK_EXECUTION_TIME,
                                    A.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
                                    A.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
                                    A.VL_COMMUNICATION_TIME_1,
                                    A.VL_COMMUNICATION_TIME_2,
                                    A.VL_COMMUNICATION_TIME_1 + A.VL_COMMUNICATION_TIME_2 AS VL_COMMUNICATION_TIME,
                                    A.VL_ADDITIONAL_TIME_1,
                                    A.VL_ADDITIONAL_TIME_2,
                                    A.VL_ADDITIONAL_TIME_1 + A.VL_ADDITIONAL_TIME_2 AS VL_ADDITIONAL_TIME,
                                    (A.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC + A.VL_ADDITIONAL_TIME_1 + A.VL_ADDITIONAL_TIME_2) AS VL_INTRA_TASK_EXECUTION_TIME_FREE_ADDITIONAL,
                                    A.DT_PROCESSING,
                                    B.ID_PARAMETER,
                                    B.CD_PARAMETER,
                                    B.CD_CONFIGURATION,
                                    B.ID_ALGORITHM,
                                    (SELECT DISTINCT X.DS_ALGORITHM FROM ALGORITHM X WHERE X.ID_ALGORITHM = B.ID_ALGORITHM) AS DS_ALGORITHM,
                                    B.ID_FUNCTION,
                                    (SELECT DISTINCT X.DS_FUNCTION FROM FUNCTION X WHERE X.ID_FUNCTION = B.ID_FUNCTION) AS DS_FUNCTION,
                                    (SELECT DISTINCT Y.ID_DEVICE FROM FUNCTION X INNER JOIN DEVICE Y ON (X.ID_DEVICE = Y.ID_DEVICE) WHERE X.ID_FUNCTION = B.ID_FUNCTION) AS ID_DEVICE,
                                    (SELECT DISTINCT Y.DS_DEVICE FROM FUNCTION X INNER JOIN DEVICE Y ON (X.ID_DEVICE = Y.ID_DEVICE) WHERE X.ID_FUNCTION = B.ID_FUNCTION) AS DS_DEVICE,
                                    B.ID_DATASET,
                                    B.ID_RESOURCE,
                                    B.ID_PARAMETER_TYPE,
                                    (SELECT X.DS_PARAMETER_TYPE FROM PARAMETER_TYPE X WHERE X.ID_PARAMETER_TYPE = B.ID_PARAMETER_TYPE) AS DS_PARAMETER_TYPE,
                                    (SELECT X.DS_PARAMETER_ATTRIBUTE FROM PARAMETER_TYPE X WHERE X.ID_PARAMETER_TYPE = B.ID_PARAMETER_TYPE) AS DS_PARAMETER_ATTRIBUTE,
                                    B.NR_ITERATIONS,
                                    B.VL_GRID_ROW_DIMENSION,
                                    B.VL_GRID_COLUMN_DIMENSION,
                                    B.VL_GRID_ROW_DIMENSION || ' x ' || B.VL_GRID_COLUMN_DIMENSION AS VL_GRID_ROW_X_COLUMN_DIMENSION,
									B.VL_GRID_ROW_DIMENSION || ' x ' || B.VL_GRID_COLUMN_DIMENSION || '(' || ROUND(B.VL_BLOCK_MEMORY_SIZE*1e-6,2) || ')' AS VL_CONCAT_GRID_ROW_X_COLUMN_DIMENSION_BLOCK_SIZE_MB,
                                    ROUND(D.VL_DATASET_MEMORY_SIZE*1e-6,0) || ' (' || ROUND((CAST(B.VL_BLOCK_MEMORY_SIZE AS NUMERIC)/CAST(D.VL_DATASET_MEMORY_SIZE AS NUMERIC))*100,2) || '%)' AS VL_CONCAT_DATASET_MB_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
									CASE
										WHEN (SELECT DISTINCT X.DS_ALGORITHM FROM ALGORITHM X WHERE X.ID_ALGORITHM = B.ID_ALGORITHM) = 'KMEANS'
											THEN ROUND(B.VL_BLOCK_MEMORY_SIZE*1e-6,2) || ' (' || B.VL_GRID_ROW_DIMENSION*5  || ')'
										WHEN (SELECT DISTINCT X.DS_ALGORITHM FROM ALGORITHM X WHERE X.ID_ALGORITHM = B.ID_ALGORITHM) = 'MATMUL_DISLIB'
											THEN ROUND(B.VL_BLOCK_MEMORY_SIZE*1e-6,2) || ' (' || B.VL_GRID_ROW_DIMENSION*B.VL_GRID_ROW_DIMENSION*B.VL_GRID_ROW_DIMENSION + B.VL_GRID_ROW_DIMENSION*B.VL_GRID_ROW_DIMENSION*(B.VL_GRID_ROW_DIMENSION-1)  || ')'
										ELSE
											'999999999999'
									END AS VL_CONCAT_BLOCK_SIZE_MB_NR_TASKS,
									ROUND(B.VL_BLOCK_MEMORY_SIZE*1e-6,0) || ' (' || B.VL_GRID_ROW_DIMENSION || ' x ' || B.VL_GRID_COLUMN_DIMENSION  || ')' AS VL_CONCAT_BLOCK_SIZE_MB_GRID_ROW_X_COLUMN_DIMENSION,
                                    ROUND((CAST(B.VL_GRID_COLUMN_DIMENSION AS NUMERIC)/CAST(D.VL_DATASET_COLUMN_DIMENSION AS NUMERIC))*100,2) AS VL_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                                    B.VL_GRID_ROW_DIMENSION || ' x ' || B.VL_GRID_COLUMN_DIMENSION || ' (' || ROUND((CAST(B.VL_GRID_COLUMN_DIMENSION AS NUMERIC)/CAST(D.VL_DATASET_COLUMN_DIMENSION AS NUMERIC))*100,2) || '%)' AS VL_CONCAT_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                                    B.VL_BLOCK_ROW_DIMENSION,
                                    B.VL_BLOCK_COLUMN_DIMENSION,
                                    B.VL_BLOCK_ROW_DIMENSION || ' x ' || B.VL_BLOCK_COLUMN_DIMENSION AS VL_BLOCK_ROW_X_COLUMN_DIMENSION,
                                    ROUND(B.VL_BLOCK_MEMORY_SIZE*1e-6,2) AS VL_BLOCK_MEMORY_SIZE,
                                    B.VL_BLOCK_MEMORY_SIZE_PERCENT_CPU,
                                    B.VL_BLOCK_MEMORY_SIZE_PERCENT_GPU,
                                    ROUND((CAST(B.VL_BLOCK_MEMORY_SIZE AS NUMERIC)/CAST(D.VL_DATASET_MEMORY_SIZE AS NUMERIC))*100,2) AS VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                                    ROUND(B.VL_BLOCK_MEMORY_SIZE*1e-6,2) || ' (' || ROUND((CAST(B.VL_BLOCK_MEMORY_SIZE AS NUMERIC)/CAST(D.VL_DATASET_MEMORY_SIZE AS NUMERIC))*100,2) || '%)' AS VL_CONCAT_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                                    C.DS_RESOURCE,
                                    C.NR_NODES,
                                    C.NR_COMPUTING_UNITS_CPU,
                                    C.NR_COMPUTING_UNITS_GPU,
                                    C.VL_MEMORY_SIZE_PER_CPU_COMPUTING_UNIT,
                                    C.VL_MEMORY_SIZE_PER_GPU_COMPUTING_UNIT,
                                    D.DS_DATASET,
                                    D.VL_DATASET_MEMORY_SIZE,
                                    D.DS_DATA_TYPE,
                                    D.VL_DATA_TYPE_MEMORY_SIZE,
                                    D.VL_DATASET_DIMENSION,
                                    D.VL_DATASET_ROW_DIMENSION,
                                    D.VL_DATASET_COLUMN_DIMENSION,
                                    D.VL_DATASET_ROW_DIMENSION || ' x ' || D.VL_DATASET_COLUMN_DIMENSION AS VL_DATASET_ROW_X_COLUMN_DIMENSION,
                                    D.NR_RANDOM_STATE,
									D.VL_DATA_SPARSITY,
									D.VL_DATA_SKEWNESS
                                FROM EXPERIMENT_RAW A
                                INNER JOIN PARAMETER B ON (A.ID_PARAMETER = B.ID_PARAMETER)
                                INNER JOIN RESOURCE C ON (B.ID_RESOURCE = C.ID_RESOURCE)
                                INNER JOIN DATASET D ON (B.ID_DATASET = D.ID_DATASET)
                                WHERE
                                --A.NR_ALGORITHM_ITERATION <> 0
                                --AND DATE_TRUNC('day', A.DT_PROCESSING) < TO_DATE('15/11/2022', 'dd/mm/yyyy')
                                --FILTERS TO REMOVE TOP 10 OUTLIERS IN SKEWNED DATASET EXPERIMENTS (WHERE JUST 1 EXECUTION WAS DONE) 
                                --AND A.ID_EXPERIMENT NOT IN () -- GRID ROW DIM 1 SKEWED ('S_1GB_1') -- NO NEED TO FILTER (ONLY ONE TASK)
                                --AND A.ID_EXPERIMENT NOT IN () -- GRID ROW DIM 1 NOT SKEWED ('S_1GB_3') -- NO NEED TO FILTER (ONLY ONE TASK)
                                A.ID_EXPERIMENT NOT IN (784797,784799,784798) -- GRID ROW DIM 2 SKEWED ('S_1GB_1')
                                AND A.ID_EXPERIMENT NOT IN (779557,779558,779559) -- GRID ROW DIM 2 NOT SKEWED ('S_1GB_3')
                                AND A.ID_EXPERIMENT NOT IN (784838,784839,784836,784845,784837,784847,784844) -- GRID ROW DIM 4 SKEWED ('S_1GB_1')
                                AND A.ID_EXPERIMENT NOT IN (779598,779599,779609,779614,779610,779597,779604) -- GRID ROW DIM 4 NOT SKEWED ('S_1GB_3')
                                AND A.ID_EXPERIMENT NOT IN (784911,784913,784912,784910,784909,784908,784944,784929,784928,784938) -- GRID ROW DIM 8 SKEWED ('S_1GB_1')
                                AND A.ID_EXPERIMENT NOT IN (779672,779671,779673,779670,779668,779669,779667,779676,779683,779677) -- GRID ROW DIM 8 NOT SKEWED ('S_1GB_3')
                                AND A.ID_EXPERIMENT NOT IN (785049,785048,785046,785047,785038,785039,785057,785095,785067,785060) -- GRID ROW DIM 16 SKEWED ('S_1GB_1')
                                AND A.ID_EXPERIMENT NOT IN (779805,779807,779806,779810,779811,779808,779809,779800,779799,779798) -- GRID ROW DIM 16 NOT SKEWED ('S_1GB_3')
                                AND A.ID_EXPERIMENT NOT IN (785321,785320,785323,785322,785310,785309,785311,785308,785317,785318) -- GRID ROW DIM 32 SKEWED ('S_1GB_1')
                                AND A.ID_EXPERIMENT NOT IN (780079,780078,780074,780075,780076,780077,780083,780082,780081,780080) -- GRID ROW DIM 32 NOT SKEWED ('S_1GB_3')
                                AND A.ID_EXPERIMENT NOT IN (785827,785832,785825,785826,785828,785834,785824,785823,786016,786080) -- GRID ROW DIM 64 SKEWED ('S_1GB_1')
                                AND A.ID_EXPERIMENT NOT IN (780603,780602,780607,780606,780601,780605,780568,780565,780589,780567) -- GRID ROW DIM 64 NOT SKEWED ('S_1GB_3')
                                AND A.ID_EXPERIMENT NOT IN (786781,786784,786780,786783,786793,786827,786812,786807,786808,786803) -- GRID ROW DIM 128 SKEWED ('S_1GB_1')
                                AND A.ID_EXPERIMENT NOT IN (781551,781549,781548,781544,781545,781541,781546,781540,781562,781542) -- GRID ROW DIM 128 NOT SKEWED ('S_1GB_3')
                                AND A.ID_EXPERIMENT NOT IN (788843,788847,788728,788729,788735,788734,788739,788738,788727,788733) -- GRID ROW DIM 256 SKEWED ('S_1GB_1')
                            ) X
                            GROUP BY
                            X.ID_PARAMETER,
                            X.CD_PARAMETER,
                            X.CD_CONFIGURATION,
                            X.ID_ALGORITHM,
                            X.DS_ALGORITHM,
                            X.ID_FUNCTION,
                            X.DS_FUNCTION,
                            X.ID_DEVICE,
                            X.DS_DEVICE,
                            X.ID_DATASET,
                            X.ID_RESOURCE,
                            X.ID_PARAMETER_TYPE,
                            X.DS_PARAMETER_TYPE,
                            X.DS_PARAMETER_ATTRIBUTE,
                            X.NR_ITERATIONS,
                            X.VL_GRID_ROW_DIMENSION,
                            X.VL_GRID_COLUMN_DIMENSION,
                            X.VL_GRID_ROW_X_COLUMN_DIMENSION,
							X.VL_CONCAT_GRID_ROW_X_COLUMN_DIMENSION_BLOCK_SIZE_MB,
                            X.VL_CONCAT_DATASET_MB_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
							X.VL_CONCAT_BLOCK_SIZE_MB_NR_TASKS,
                            X.VL_CONCAT_BLOCK_SIZE_MB_GRID_ROW_X_COLUMN_DIMENSION,
                            X.VL_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                            X.VL_CONCAT_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                            X.VL_BLOCK_ROW_DIMENSION,
                            X.VL_BLOCK_COLUMN_DIMENSION,
                            X.VL_BLOCK_ROW_X_COLUMN_DIMENSION,
							X.VL_BLOCK_MEMORY_SIZE,
                            X.VL_BLOCK_MEMORY_SIZE_PERCENT_CPU,
                            X.VL_BLOCK_MEMORY_SIZE_PERCENT_GPU,
                            X.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                            X.VL_CONCAT_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                            X.DS_RESOURCE,
                            X.NR_NODES,
                            X.NR_COMPUTING_UNITS_CPU,
                            X.NR_COMPUTING_UNITS_GPU,
                            X.VL_MEMORY_SIZE_PER_CPU_COMPUTING_UNIT,
                            X.VL_MEMORY_SIZE_PER_GPU_COMPUTING_UNIT,
                            X.DS_DATASET,
                            X.VL_DATASET_MEMORY_SIZE,
                            X.DS_DATA_TYPE,
                            X.VL_DATA_TYPE_MEMORY_SIZE,
                            X.VL_DATASET_DIMENSION,
                            X.VL_DATASET_ROW_DIMENSION,
                            X.VL_DATASET_COLUMN_DIMENSION,
                            X.VL_DATASET_ROW_X_COLUMN_DIMENSION,
                            X.NR_RANDOM_STATE,
							X.VL_DATA_SPARSITY,
							X.VL_DATA_SKEWNESS
                        ) Y
                        ORDER BY ID_PARAMETER;