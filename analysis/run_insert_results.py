import os
import csv
import psycopg2
import psycopg2.extras as extras
import pandas as pd
import math
from db_config.config import open_connection, close_connection

def main():

    # Path of the "tb_experiments_raw" table - CSV file (raw log metrics)
    src_path_experiments = "results/tb_experiments_raw.csv"

    # Path of the "tb_experiments" table - CSV file (prepared metrics)
    dst_path_experiments = "results/tb_experiments.csv"

    # Reading "tb_experiments_raw" table
    param_file = os.path.join(src_path_experiments)
    df_parameters = pd.read_csv(param_file)

    # defining the structure of "tb_experiments" table
    header = read_column_names_csv(dst_path_experiments)[0]

    # open "tb_experiments" in write mode
    f = open(dst_path_experiments, "w", encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(header)
    f.close()

    for index, row in df_parameters.iterrows():

        # Treating null values
        id_parameter = row["id_parameter"]
        nr_algorithm_iteration = row["nr_algorithm_iteration"]
        nr_function_iteration = row["nr_function_iteration"]
        nr_task = None if math.isnan(row["nr_task"]) else row["nr_task"]
        vl_total_execution_time = None if (math.isnan(row["start_total_execution_time"] or row["end_total_execution_time"])) else row["end_total_execution_time"] - row["start_total_execution_time"]
        vl_inter_task_execution_time = None if (math.isnan(row["start_inter_time_cpu"] or row["end_inter_time_cpu"])) else row["end_inter_time_cpu"] - row["start_inter_time_cpu"]
        vl_intra_task_execution_time_device_func = None if math.isnan(row["vl_intra_task_execution_time_device_func"]) else row["vl_intra_task_execution_time_device_func"]        
        vl_communication_time_1 = None if (math.isnan(row["start_communication_time_1"] or row["end_communication_time_1"])) else row["end_communication_time_1"] - row["start_communication_time_1"]
        vl_communication_time_2 = None if (math.isnan(row["start_communication_time_2"] or row["end_communication_time_2"])) else row["end_communication_time_2"] - row["start_communication_time_2"]
        vl_additional_time_1 = None if (math.isnan(row["start_additional_time_1"] or row["end_additional_time_1"])) else row["end_additional_time_1"] - row["start_additional_time_1"]
        vl_additional_time_2 = None if (math.isnan(row["start_additional_time_2"] or row["end_additional_time_2"])) else row["end_additional_time_2"] - row["start_additional_time_2"]
        vl_intra_task_execution_time_full_func = None if (vl_communication_time_1 is None or vl_communication_time_2 is None or vl_intra_task_execution_time_device_func is None or vl_additional_time_1 is None or vl_additional_time_2 is None) else vl_communication_time_1 + vl_communication_time_2 + vl_intra_task_execution_time_device_func + vl_additional_time_1 + vl_additional_time_2
        dt_processing = row["dt_processing"]

        # open "tb_experiments" in append mode
        f = open(dst_path_experiments, "a", encoding='UTF8', newline='')

        # create a csv writer
        writer = csv.writer(f)

        data = [id_parameter, nr_algorithm_iteration, nr_function_iteration, nr_task, vl_total_execution_time, vl_inter_task_execution_time, vl_intra_task_execution_time_full_func, vl_intra_task_execution_time_device_func, vl_communication_time_1, vl_communication_time_2, vl_additional_time_1, vl_additional_time_2, dt_processing]
        writer.writerow(data)
        f.close()

    # Reading "tb_experiments" table
    param_file = os.path.join(dst_path_experiments)
    df_experiments = pd.read_csv(param_file)

    tuples = [tuple(x) for x in df_experiments.to_numpy()]
    
    cols = ','.join(list(df_experiments.columns))

    # Open connection to the database
    schema='schema_kmeans'
    cur, conn = open_connection(schema)

    sql_create_schema = "SET search_path = "+schema+";"
    cur.execute(sql_create_schema)

    # Set sql query - on conflict with the database values, do nothing
    sql_query = "INSERT INTO EXPERIMENT_RAW(%s) VALUES %%s ON CONFLICT (%s) DO NOTHING" % (cols,cols)

    new_tuples = [tuple(None if isinstance(i, float) and math.isnan(i) else i for i in t) for t in tuples]

    # Get dataframe from query
    insert_experiment_result(sql_query, cur, conn, new_tuples)

# Function that takes in a PostgreSQL query and outputs a pandas dataframe 
def insert_experiment_result(sql_query, cur, conn, tuples):
    try:
        extras.execute_values(cur, sql_query, tuples)
        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        conn.rollback()
        close_connection(cur, conn)

    print("Values inserted successfully!")
    close_connection(cur, conn)

def read_column_names_csv(dst_path_experiments):
    with open(dst_path_experiments) as csv_file:
 
        # creating an object of csv reader with the delimiter as ,
        csv_reader = csv.reader(csv_file, delimiter = ',')
    
        # list to store the names of columns
        list_of_column_names = []
    
        # loop to iterate through the rows of csv
        for row in csv_reader:
    
            # adding the first row
            list_of_column_names.append(row)
    
            # breaking the loop after the first iteration itself
            break
    return list_of_column_names

if __name__ == "__main__":
    main()