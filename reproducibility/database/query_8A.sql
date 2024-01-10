WITH T_CPU AS (
                                    SELECT
                                    Y.VL_TOTAL_EXECUTION_TIME,
                                    Y.VL_INTER_TASK_EXECUTION_TIME,
									Y.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
									Y.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
									Y.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC - (Y.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC+Y.VL_COMMUNICATION_TIME) AS VL_ADDITIONAL_TIME,
									Y.VL_COMMUNICATION_TIME,
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
                                    Y.VL_BLOCK_ROW_DIMENSION,
                                    Y.VL_BLOCK_COLUMN_DIMENSION,
                                    Y.VL_BLOCK_MEMORY_SIZE,
                                    Y.VL_BLOCK_MEMORY_SIZE_PERCENT_CPU,
                                    Y.VL_BLOCK_MEMORY_SIZE_PERCENT_GPU,
                                    Y.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                                    Y.DS_RESOURCE,
                                    Y.NR_NODES,
                                    Y.NR_COMPUTING_UNITS_CPU,
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
                                    Y.NR_RANDOM_STATE
                                    FROM
                                    (
                                        SELECT
                                        AVG(X.VL_TOTAL_EXECUTION_TIME) AS VL_TOTAL_EXECUTION_TIME,
                                        AVG(X.VL_INTER_TASK_EXECUTION_TIME) AS VL_INTER_TASK_EXECUTION_TIME,
                                        AVG(X.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC) AS VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
                                        AVG(X.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC) AS VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
                                        AVG(X.VL_COMMUNICATION_TIME_1) AS VL_COMMUNICATION_TIME_1,
                                        AVG(X.VL_COMMUNICATION_TIME_2) AS VL_COMMUNICATION_TIME_2,
                                        AVG(X.VL_COMMUNICATION_TIME) AS VL_COMMUNICATION_TIME,
                                        AVG(X.VL_ADDITIONAL_TIME_1) AS VL_ADDITIONAL_TIME_1,
                                        AVG(X.VL_ADDITIONAL_TIME_2) AS VL_ADDITIONAL_TIME_2,
                                        AVG(X.VL_ADDITIONAL_TIME) AS VL_ADDITIONAL_TIME,
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
                                        X.NR_RANDOM_STATE
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
                                                ROUND((CAST(B.VL_GRID_COLUMN_DIMENSION AS NUMERIC)/CAST(D.VL_DATASET_COLUMN_DIMENSION AS NUMERIC))*100,2) AS VL_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                                                B.VL_GRID_ROW_DIMENSION || ' x ' || B.VL_GRID_COLUMN_DIMENSION || ' (' || ROUND((CAST(B.VL_GRID_COLUMN_DIMENSION AS NUMERIC)/CAST(D.VL_DATASET_COLUMN_DIMENSION AS NUMERIC))*100,2) || '%)' AS VL_CONCAT_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                                                B.VL_BLOCK_ROW_DIMENSION,
                                                B.VL_BLOCK_COLUMN_DIMENSION,
                                                B.VL_BLOCK_ROW_DIMENSION || ' x ' || B.VL_BLOCK_COLUMN_DIMENSION AS VL_BLOCK_ROW_X_COLUMN_DIMENSION,
                                                B.VL_BLOCK_MEMORY_SIZE,
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
                                                D.NR_RANDOM_STATE
                                            FROM EXPERIMENT_RAW A
                                            INNER JOIN PARAMETER B ON (A.ID_PARAMETER = B.ID_PARAMETER)
                                            INNER JOIN RESOURCE C ON (B.ID_RESOURCE = C.ID_RESOURCE)
                                            INNER JOIN DATASET D ON (B.ID_DATASET = D.ID_DATASET)
                                            WHERE
                                            (SELECT DISTINCT Z.DS_DEVICE FROM FUNCTION W INNER JOIN DEVICE Z ON (W.ID_DEVICE = Z.ID_DEVICE) WHERE W.ID_FUNCTION = B.ID_FUNCTION) = 'CPU'
                                            AND A.NR_ALGORITHM_ITERATION <> 0
                                        ) X
										WHERE
										X.DS_FUNCTION = 'ADD_FUNC'
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
                                        X.NR_RANDOM_STATE
                                    ) Y

                                ),
                    T_GPU AS (
                                    SELECT
                                    Y.VL_TOTAL_EXECUTION_TIME,
                                    Y.VL_INTER_TASK_EXECUTION_TIME,
                                    Y.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
									Y.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
									Y.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC - (Y.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC+Y.VL_COMMUNICATION_TIME) AS VL_ADDITIONAL_TIME,
									Y.VL_COMMUNICATION_TIME,
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
                                    Y.VL_BLOCK_ROW_DIMENSION,
                                    Y.VL_BLOCK_COLUMN_DIMENSION,
                                    Y.VL_BLOCK_MEMORY_SIZE,
                                    Y.VL_BLOCK_MEMORY_SIZE_PERCENT_CPU,
                                    Y.VL_BLOCK_MEMORY_SIZE_PERCENT_GPU,
                                    Y.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                                    Y.DS_RESOURCE,
                                    Y.NR_NODES,
                                    Y.NR_COMPUTING_UNITS_CPU,
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
                                    Y.NR_RANDOM_STATE
                                    FROM
                                    (
                                        SELECT
                                        AVG(X.VL_TOTAL_EXECUTION_TIME) AS VL_TOTAL_EXECUTION_TIME,
                                        AVG(X.VL_INTER_TASK_EXECUTION_TIME) AS VL_INTER_TASK_EXECUTION_TIME,
                                        AVG(X.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC) AS VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC,
                                        AVG(X.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC) AS VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC,
                                        AVG(X.VL_COMMUNICATION_TIME_1) AS VL_COMMUNICATION_TIME_1,
                                        AVG(X.VL_COMMUNICATION_TIME_2) AS VL_COMMUNICATION_TIME_2,
                                        AVG(X.VL_COMMUNICATION_TIME) AS VL_COMMUNICATION_TIME,
                                        AVG(X.VL_ADDITIONAL_TIME_1) AS VL_ADDITIONAL_TIME_1,
                                        AVG(X.VL_ADDITIONAL_TIME_2) AS VL_ADDITIONAL_TIME_2,
                                        AVG(X.VL_ADDITIONAL_TIME) AS VL_ADDITIONAL_TIME,
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
                                        X.NR_RANDOM_STATE
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
                                                ROUND((CAST(B.VL_GRID_COLUMN_DIMENSION AS NUMERIC)/CAST(D.VL_DATASET_COLUMN_DIMENSION AS NUMERIC))*100,2) AS VL_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                                                B.VL_GRID_ROW_DIMENSION || ' x ' || B.VL_GRID_COLUMN_DIMENSION || ' (' || ROUND((CAST(B.VL_GRID_COLUMN_DIMENSION AS NUMERIC)/CAST(D.VL_DATASET_COLUMN_DIMENSION AS NUMERIC))*100,2) || '%)' AS VL_CONCAT_GRID_COLUMN_DIMENSION_PERCENT_DATASET,
                                                B.VL_BLOCK_ROW_DIMENSION,
                                                B.VL_BLOCK_COLUMN_DIMENSION,
                                                B.VL_BLOCK_ROW_DIMENSION || ' x ' || B.VL_BLOCK_COLUMN_DIMENSION AS VL_BLOCK_ROW_X_COLUMN_DIMENSION,
                                                B.VL_BLOCK_MEMORY_SIZE,
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
                                                D.NR_RANDOM_STATE
                                            FROM EXPERIMENT_RAW A
                                            INNER JOIN PARAMETER B ON (A.ID_PARAMETER = B.ID_PARAMETER)
                                            INNER JOIN RESOURCE C ON (B.ID_RESOURCE = C.ID_RESOURCE)
                                            INNER JOIN DATASET D ON (B.ID_DATASET = D.ID_DATASET)
                                            WHERE
                                            (SELECT DISTINCT Z.DS_DEVICE FROM FUNCTION W INNER JOIN DEVICE Z ON (W.ID_DEVICE = Z.ID_DEVICE) WHERE W.ID_FUNCTION = B.ID_FUNCTION) = 'GPU'
                                            AND A.NR_ALGORITHM_ITERATION <> 0
                                        ) X
										WHERE
										X.DS_FUNCTION = 'ADD_FUNC'
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
                                        X.NR_RANDOM_STATE
                                    ) Y
                    )
                    SELECT
                    T_CPU.ID_PARAMETER_TYPE,
                    T_CPU.CD_PARAMETER,
                    T_CPU.DS_ALGORITHM,
                    T_CPU.DS_FUNCTION,
                    T_CPU.NR_ITERATIONS,
                    CASE
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 0.3 THEN 0.4
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 0.7 THEN 0.8
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 1.5 THEN 1.6
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 3.0 THEN 3.1
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 6.2 THEN 6.3
                    ELSE ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) END AS VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                    CASE
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 0.3 THEN 0.4 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION || ' x ' || T_CPU.VL_GRID_COLUMN_DIMENSION  || ')'
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 0.7 THEN 0.8 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION || ' x ' || T_CPU.VL_GRID_COLUMN_DIMENSION  || ')'
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 1.5 THEN 1.6 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION || ' x ' || T_CPU.VL_GRID_COLUMN_DIMENSION  || ')'
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 3.0 THEN 3.1 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION || ' x ' || T_CPU.VL_GRID_COLUMN_DIMENSION  || ')'
                    WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 6.2 THEN 6.3 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION || ' x ' || T_CPU.VL_GRID_COLUMN_DIMENSION  || ')'
                    ELSE ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) || ' (' || T_CPU.VL_GRID_ROW_DIMENSION || ' x ' || T_CPU.VL_GRID_COLUMN_DIMENSION  || ')' END AS CONCAT_BLOCK_PERCENT_DATASET_GRID_DIMENSION,
                    --ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) AS VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,
                    T_CPU.DS_RESOURCE,
                    T_CPU.DS_PARAMETER_TYPE,
                    T_CPU.DS_PARAMETER_ATTRIBUTE,
                    T_CPU.DS_DATASET,
                    CAST(T_CPU.VL_DATASET_MEMORY_SIZE*1e-6 AS BIGINT) as VL_DATASET_MEMORY_SIZE,
                    T_CPU.VL_DATASET_DIMENSION,
                    T_CPU.VL_DATASET_ROW_DIMENSION,
                    T_CPU.VL_DATASET_COLUMN_DIMENSION,
                    T_CPU.VL_GRID_ROW_DIMENSION,
                    T_CPU.VL_GRID_COLUMN_DIMENSION,
                    T_CPU.VL_BLOCK_ROW_DIMENSION,
                    T_CPU.VL_BLOCK_COLUMN_DIMENSION,
                    T_CPU.VL_BLOCK_MEMORY_SIZE,
                    ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE*1e-6,0)::varchar(255) as VL_BLOCK_MEMORY_SIZE_MB,
                    ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE*1e-6,0) || ' (' || T_CPU.VL_GRID_ROW_DIMENSION || ' x ' || T_CPU.VL_GRID_COLUMN_DIMENSION  || ')' AS VL_CONCAT_BLOCK_SIZE_MB_GRID_ROW_X_COLUMN_DIMENSION,
					CASE
						WHEN T_CPU.DS_ALGORITHM = 'KMEANS'
							THEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE*1e-6,0) || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*5 || ')'
						WHEN T_CPU.DS_ALGORITHM = 'MATMUL_DISLIB'
							THEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE*1e-6,0) || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION + T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*(T_CPU.VL_GRID_ROW_DIMENSION-1)  || ')'
						ELSE
							'999999999999'
					END AS VL_CONCAT_BLOCK_SIZE_MB_NR_TASKS,
					CASE
						WHEN T_CPU.DS_ALGORITHM = 'KMEANS'
							THEN
								CASE
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 0.3 THEN 0.4 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*5  || ')'
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 0.7 THEN 0.8 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*5  || ')'
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 1.5 THEN 1.6 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*5  || ')'
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 3.0 THEN 3.1 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*5  || ')'
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 6.2 THEN 6.3 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*5  || ')'
									ELSE ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*5  || ')'
								END
						WHEN T_CPU.DS_ALGORITHM = 'MATMUL_DISLIB'
							THEN
								CASE
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 0.3 THEN 0.4 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION + T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*(T_CPU.VL_GRID_ROW_DIMENSION-1)  || ')'
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 0.7 THEN 0.8 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION + T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*(T_CPU.VL_GRID_ROW_DIMENSION-1)  || ')'
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 1.5 THEN 1.6 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION + T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*(T_CPU.VL_GRID_ROW_DIMENSION-1)  || ')'
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 3.0 THEN 3.1 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION + T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*(T_CPU.VL_GRID_ROW_DIMENSION-1)  || ')'
									WHEN ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) = 6.2 THEN 6.3 || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION + T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*(T_CPU.VL_GRID_ROW_DIMENSION-1)  || ')'
									ELSE ROUND(T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET,1) || ' (' || T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION + T_CPU.VL_GRID_ROW_DIMENSION*T_CPU.VL_GRID_ROW_DIMENSION*(T_CPU.VL_GRID_ROW_DIMENSION-1)  || ')'
								END
						ELSE
							'999999999999'
					END AS CONCAT_BLOCK_PERCENT_DATASET_NR_TASKS,
					T_CPU.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC AS VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC_CPU,
					T_GPU.VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC AS VL_INTRA_TASK_EXECUTION_TIME_DEVICE_FUNC_GPU,
					T_CPU.VL_ADDITIONAL_TIME AS VL_ADDITIONAL_TIME_CPU,
					T_GPU.VL_ADDITIONAL_TIME AS VL_ADDITIONAL_TIME_GPU,
					T_CPU.VL_COMMUNICATION_TIME AS VL_COMMUNICATION_TIME_CPU,
					T_GPU.VL_COMMUNICATION_TIME AS VL_COMMUNICATION_TIME_GPU,
                    CASE
                    WHEN ROUND((T_CPU.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC/T_GPU.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC)::numeric,2) > 1.00 THEN ROUND((T_CPU.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC/T_GPU.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC)::numeric,2)
                    ELSE -ROUND((T_GPU.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC/T_CPU.VL_INTRA_TASK_EXECUTION_TIME_FULL_FUNC)::numeric,2)
                    END AS SPEEDUP_GPU_INTRA_TASK_EXECUTION_TIME_FULL_FUNC
                    FROM T_CPU LEFT JOIN T_GPU ON (T_CPU.CD_PARAMETER = T_GPU.CD_PARAMETER
													AND T_CPU.VL_GRID_ROW_DIMENSION = T_GPU.VL_GRID_ROW_DIMENSION
													AND T_CPU.VL_GRID_COLUMN_DIMENSION = T_GPU.VL_GRID_COLUMN_DIMENSION
													AND T_CPU.VL_BLOCK_ROW_DIMENSION = T_GPU.VL_BLOCK_ROW_DIMENSION
													AND T_CPU.VL_BLOCK_COLUMN_DIMENSION = T_GPU.VL_BLOCK_COLUMN_DIMENSION)
                    ORDER BY
                    T_CPU.DS_DATASET,
                    T_CPU.VL_BLOCK_MEMORY_SIZE_PERCENT_DATASET;