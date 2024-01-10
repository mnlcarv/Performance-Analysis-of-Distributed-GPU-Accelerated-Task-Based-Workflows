import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from db_config.config import open_connection, close_connection


# Main function to plot charts (usage python create_plots.py)
def main():

    # Setup database: create schemas, create tables, load tables
    setup_database()

    path_queries = "/database/"
    path_csv_file = "/raw_data/combined_data/" # FOR CHARTS THAT DEPEND ON CROSSING DATA WITH PARAVER

    queries = get_files_with_prefix(path_queries,'query')

    dst_path_figs = '/figures/'

    for query in queries:
        
        if (query == 'query_7.sql'):

            # Reading crossing data from csv file: database (query_7.sql) with paraver (raw_data/combined_data/paraver_metrics) 
            dst_combined_data = path_csv_file+"tb_cross_data_paraver.csv"

            # Reading "tb_experiments_motivation" csv table
            param_file = os.path.join(dst_combined_data)
            df_filtered = pd.read_csv(param_file)

            df = df_filtered

            generate_graph(df, dst_path_figs, query)
        
        elif (query == 'query_8.sql'):

            # Open connection to the database
            schema='schema_matmul'
            cur, conn = open_connection(schema)

            # Read query
            sql_file_pathA = "/database/query_8A.sql"
            sql_queryA = read_sql_file(sql_file_pathA)
            sql_file_pathB = "/database/query_8B.sql"
            sql_queryB = read_sql_file(sql_file_pathB)

            # Get dataframe from query
            dfA = get_df_from_query(sql_queryA,conn)
            dfB = get_df_from_query(sql_queryB,conn)
            df = pd.concat([dfA, dfB])

            generate_graph(df, dst_path_figs, query)

            # Close connection to the database
            close_connection(cur, conn)

        elif (query == 'query_9A.sql'):

            # Open connection to the database
            schema='schema_kmeans'
            cur, conn = open_connection(schema)
            
            # Read query
            sql_file_path = "/database/"+query
            sql_query = read_sql_file(sql_file_path)

            # Get dataframe from query
            df = get_df_from_query(sql_query,conn)

            generate_graph(df, dst_path_figs, query)

            # Close connection to the database
            close_connection(cur, conn)

        elif (query == 'query_9B1.sql'):

            # Open connection to the database
            schema='schema_matmul'
            cur, conn = open_connection(schema)
           
            # Read query
            sql_file_path = "/database/"+query
            sql_query = read_sql_file(sql_file_path)

            # Get dataframe from query
            df = get_df_from_query(sql_query,conn)

            generate_graph(df, dst_path_figs, query)

            # Close connection to the database
            close_connection(cur, conn)

        elif (query == 'query_9B2.sql'):
            
            # Open connection to the database
            schema='schema_kmeans'
            cur, conn = open_connection(schema)

            # Read query
            sql_file_path = "/database/"+query
            sql_query = read_sql_file(sql_file_path)

            # Get dataframe from query
            df = get_df_from_query(sql_query,conn)

            generate_graph(df, dst_path_figs, query)

            # Close connection to the database
            close_connection(cur, conn)

        elif (query == 'query_10A.sql'):
            
            # Open connection to the database
            schema='schema_matmul'
            cur, conn = open_connection(schema)

            # Read query
            sql_file_path = "/database/"+query
            sql_query = read_sql_file(sql_file_path)

            # Get dataframe from query
            df = get_df_from_query(sql_query,conn)

            generate_graph(df, dst_path_figs, query)

            # Close connection to the database
            close_connection(cur, conn)


        elif (query == 'query_10B.sql'):
            
            # Open connection to the database
            schema='schema_kmeans'
            cur, conn = open_connection(schema)

            # Read query
            sql_file_path = "/database/"+query
            sql_query = read_sql_file(sql_file_path)

            # Get dataframe from query
            df = get_df_from_query(sql_query,conn)

            generate_graph(df, dst_path_figs, query)

            # Close connection to the database
            close_connection(cur, conn)

        elif (query == 'query_11.sql'):
            
            # Reading data combined from the results of queries in query_11.sql
            dst_combined_data = path_csv_file+"tb_correlation.csv"

            # Reading "tb_experiments_motivation" csv table
            param_file = os.path.join(dst_combined_data)
            df_filtered = pd.read_csv(param_file)

            df = df_filtered

            generate_graph(df, dst_path_figs, query)
            

        elif (query == 'query_12.sql'):
            
            # Open connection to the database
            schema='schema_matmul_fma'
            cur, conn = open_connection(schema)

            # Read query
            sql_file_path = "/database/"+query
            sql_query = read_sql_file(sql_file_path)

            # Get dataframe from query
            df = get_df_from_query(sql_query,conn)

            generate_graph(df, dst_path_figs, query)

            # Close connection to the database
            close_connection(cur, conn)

        elif (query == 'query_13.sql'):
            
            # Open connection to the database
            schema='schema_matmul'
            cur, conn = open_connection(schema)

            # Read query
            sql_file_path = "/database/"+query
            sql_query = read_sql_file(sql_file_path)

            # Get dataframe from query
            df = get_df_from_query(sql_query,conn)

            generate_graph(df, dst_path_figs, query)

            # Close connection to the database
            close_connection(cur, conn)

        else:
            print('error: invalid query')




def generate_graph(df, dst_path_figs, query):


    if query == 'query_7.sql':
        
        matplotlib.rcParams.update({'font.size': 18})

        # Matmul Speedup
        # 8 GB 
        df_filtered_left_top = df_filtered[(df_filtered.ds_algorithm=='Matmul') & (df_filtered.ds_dataset=='S_8GB_1')]
        # 32 GB
        df_filtered_right_top = df_filtered[(df_filtered.ds_algorithm=='Matmul') & (df_filtered.ds_dataset=='S_32GB_1')]
        # Matmul Time
        # 8 GB 
        df_filtered_left_bottom = df_filtered[(df_filtered.ds_algorithm=='Matmul') & (df_filtered.ds_dataset=='S_8GB_1')]
        # 32 GB
        df_filtered_right_bottom = df_filtered[(df_filtered.ds_algorithm=='Matmul') & (df_filtered.ds_dataset=='S_32GB_1')]
        left_title = 'Dataset size 8 GB'
        right_title = 'Dataset size 32 GB'

        # Speedups
        df1 = df_filtered_left_top[["concat_grid_row_x_column_dim_block_size","speedup_gpu_intra_task_execution_time_device_func","speedup_gpu_intra_task_execution_time_full_func","speedup_gpu_total_execution_time"]]
        df2 = df_filtered_right_top[["concat_grid_row_x_column_dim_block_size","speedup_gpu_intra_task_execution_time_device_func","speedup_gpu_intra_task_execution_time_full_func","speedup_gpu_total_execution_time"]]
        #Times
        df3 = df_filtered_left_bottom[["concat_grid_row_x_column_dim_block_size","vl_intra_task_execution_time_device_func","vl_intra_task_overhead","vl_inter_overhead"]]
        df4 = df_filtered_right_bottom[["concat_grid_row_x_column_dim_block_size","vl_intra_task_execution_time_device_func","vl_intra_task_overhead","vl_inter_overhead"]]

        xlabels_1 = df_filtered_left_top["concat_grid_row_x_column_dim_block_size"].drop_duplicates()
        X_axis_1 = np.arange(len(df_filtered_left_top["concat_grid_row_x_column_dim_block_size"].drop_duplicates()))

        xlabels_2 = df_filtered_right_top["concat_grid_row_x_column_dim_block_size"].drop_duplicates()
        X_axis_2 = np.arange(len(df_filtered_right_top["concat_grid_row_x_column_dim_block_size"].drop_duplicates()))
        x = 'concat_grid_row_x_column_dim_block_size'

        # Create a figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey='row', sharex='col')

        # Plot the first chart (top-left - Bar chart)
        axs[0, 0].bar(X_axis_1 - 0.3, df1['speedup_gpu_intra_task_execution_time_device_func'], 0.3, label='P. Frac.', color='C2', alpha = 0.5)
        axs[0, 0].bar(X_axis_1 + 0.0, df1['speedup_gpu_intra_task_execution_time_full_func'], 0.3, label='Usr. Code', color='C0', alpha = 0.5, hatch='///')
        axs[0, 0].bar(X_axis_1 + 0.3, df1['speedup_gpu_total_execution_time'], 0.3, label='P. Task', color='C1', alpha = 0.5, hatch='\\\\\\')
        axs[0, 0].legend(loc=(-0.000,0.99), frameon=False, labelspacing=0.01, ncol=3, borderpad=0.1)
        axs[0, 0].set_ylabel('GPU Speedup over CPU')  # Add y-axis label
        #MATMUL
        axs[0, 1].set_ylim([-5, 30])
        #KMEANS
        axs[0, 0].grid(zorder=0,axis='y')
        axs[0, 0].set_title(left_title, pad=25)

        # Plot the second chart (top-right - Bar chart)
        axs[0, 1].bar(X_axis_2 - 0.3, df2['speedup_gpu_intra_task_execution_time_device_func'], 0.3, color='C2', alpha = 0.5)
        axs[0, 1].bar(X_axis_2 + 0.0, df2['speedup_gpu_intra_task_execution_time_full_func'], 0.3, color='C0', alpha = 0.5, hatch='///')
        axs[0, 1].bar(X_axis_2 + 0.3, df2['speedup_gpu_total_execution_time'], 0.3, color='C1', alpha = 0.5, hatch='\\\\\\')
        #MATMUL
        axs[0, 1].set_ylim([-5, 30])
        #KMEANS
        axs[0, 1].grid(zorder=0,axis='y')
        axs[0, 1].set_title(right_title, pad=25)

        # Plot the third chart (bottom-left - Line chart)
        axs[1, 0].plot(df3[x], df3['vl_intra_task_execution_time_device_func'], color='C2', linestyle = '-', label='P. Frac.', linewidth=2.5)
        #MATMUL
        axs[1, 0].plot(df3[x], df3['vl_intra_task_overhead'], color='C0', linestyle = '-.', label='CPU-GPU Comm.', zorder=3, linewidth=2.5)
        #KMEANS
        axs[1, 0].plot(df3[x], df3['vl_inter_overhead'], color='C1', linestyle = '--', label='(De-)Serializ.', zorder=3, linewidth=2.5)
        axs[1, 0].legend(loc=(-0.000,0.99), frameon=False, labelspacing=0.01, ncol=3, borderpad=0.1)
        axs[1, 0].set_xlabel('Block size MB (Grid Dimension)')  # Add x-axis label
        axs[1, 0].set_ylabel('Average Exec. Time (s)')  # Add y-axis label
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_ylim([1e-2, 1e4])
        axs[1, 0].set_xticklabels(xlabels_1, rotation=30, ha='right')
        axs[1, 0].grid(zorder=0,axis='y')
        
        # Plot the fourth chart (bottom-right - Line chart)
        axs[1, 1].plot(df4[x], df4['vl_intra_task_execution_time_device_func'], color='C2', linestyle = '-', label='P. Frac.', linewidth=2.5)
        #MATMUL
        axs[1, 1].plot(df4[x], df4['vl_intra_task_overhead'], color='C0', linestyle = '-.', label='CPU-GPU Comm.', zorder=3, linewidth=2.5)
        #KMEANS
        axs[1, 1].plot(df4[x], df4['vl_inter_overhead'], color='C1', linestyle = '--', label='(De-)Serial.', zorder=3, linewidth=2.5)
        axs[1, 1].set_xlabel('Block size MB (Grid Dimension)')  # Add x-axis label
        axs[1, 1].set_yscale('log')
        axs[1, 1].set_ylim([1e-2, 1e4])
        axs[1, 1].set_xticklabels(xlabels_2, rotation=30, ha='right')
        axs[1, 1].grid(zorder=0,axis='y')

        # Adjust layout to prevent clipping of titles and labels
        plt.tight_layout()

        # Adjust spacing
        plt.subplots_adjust(wspace=0.01, hspace=0.3)
        
        # Adjust x-axis labels
        fig.autofmt_xdate(rotation=30, ha='right')

        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)
    
    elif query == 'query_8.sql':
        
        # Filtering query
        df_filtered = df[
                        (df["ds_resource"] == "MINOTAURO_9_NODES_1_CORE")
                        & (df["ds_dataset"].isin(["S_2GB_1","S_2GB_3"]))
                        & (df["ds_parameter_type"] == "VAR_GRID_SHAPE_MATMUL_1")
                        ]
        
        # Start Plot
        matplotlib.rcParams.update({'font.size': 18})

        # Matmul Speedup
        # matmul_func
        df_filtered_left_top = df_filtered[(df_filtered.ds_function=='MATMUL_FUNC')]
        # add_func
        df_filtered_right_top = df_filtered[(df_filtered.ds_function=='ADD_FUNC')]
        
        # Matmul Time
        # matmul_func
        df_filtered_left_bottom = df_filtered[(df_filtered.ds_function=='MATMUL_FUNC')]
        # add_func
        df_filtered_right_bottom = df_filtered[(df_filtered.ds_function=='ADD_FUNC')]
        left_title = 'matmul_func'
        right_title = 'add_func'

        # Speedups
        df1 = df_filtered_left_top[["vl_block_memory_size","vl_block_memory_size_mb","speedup_gpu_intra_task_execution_time_full_func"]].sort_values(by=["vl_block_memory_size"], ascending=[True])
        df2 = df_filtered_right_top[["vl_block_memory_size","vl_block_memory_size_mb","speedup_gpu_intra_task_execution_time_full_func"]].sort_values(by=["vl_block_memory_size"], ascending=[True])
        print(df_filtered_right_top)
        # Times
        df3 = df_filtered_left_bottom[["vl_block_memory_size","vl_block_memory_size_mb","vl_intra_task_execution_time_device_func_cpu","vl_additional_time_cpu","vl_communication_time_cpu","vl_intra_task_execution_time_device_func_gpu","vl_additional_time_gpu","vl_communication_time_gpu"]].sort_values(by=["vl_block_memory_size"], ascending=[True])
        df4 = df_filtered_right_bottom[["vl_block_memory_size","vl_block_memory_size_mb","vl_intra_task_execution_time_device_func_cpu","vl_additional_time_cpu","vl_communication_time_cpu","vl_intra_task_execution_time_device_func_gpu","vl_additional_time_gpu","vl_communication_time_gpu"]].sort_values(by=["vl_block_memory_size"], ascending=[True])

        x = 'vl_block_memory_size_mb'
        x_value = "vl_block_memory_size_mb"

        fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey='row', sharex='col')

        # Plot the first chart (top-left - Bar chart)
        axs[0, 0].bar(df1[x_value],df1['speedup_gpu_intra_task_execution_time_full_func'], color='C0', alpha = 0.5, label='Usr. Code')
        axs[0, 0].legend(loc=(-0.000,0.99), frameon=False, labelspacing=0.01, ncol=3, borderpad=0.1)
        axs[0, 0].set_ylabel('GPU Speedup over CPU')  # Add y-axis label
        axs[0, 0].set_ylim([-5, 25])
        axs[0, 0].grid(zorder=0,axis='y')
        axs[0, 0].set_title(left_title, pad=25, style='italic')

        # Plot the second chart (top-right - Bar chart)
        axs[0, 1].bar(df2[x],df2['speedup_gpu_intra_task_execution_time_full_func'],color='C0', alpha = 0.5)
        axs[0, 1].set_ylim([-5, 25])
        axs[0, 1].grid(zorder=0,axis='y')
        axs[0, 1].set_title(right_title, pad=25, style='italic')

        # Plot the third chart (bottom-left - Line chart)
        axs[1, 0].plot(df3[x], df3['vl_intra_task_execution_time_device_func_cpu'], color='C2', linestyle = 'dotted', label='P. Frac. CPU', linewidth=2.5)
        axs[1, 0].plot(df3[x], df3['vl_intra_task_execution_time_device_func_gpu'], color='C2', linestyle = 'solid', label='P. Frac. GPU', linewidth=2.5)
        axs[1, 0].plot(df3[x], df3['vl_communication_time_gpu'], color='C4', linestyle = 'solid', label='CPU-GPU Comm.', linewidth=2.5)
        axs[1, 0].legend(loc=(-0.000,0.99), frameon=False, labelspacing=0.01, ncol=3, borderpad=0.1)
        axs[1, 0].set_xlabel('Block size MB')  # Add x-axis label
        axs[1, 0].set_ylabel('Average Time per Task (s)')  # Add y-axis label
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_ylim([1e-3, 1e4])
        axs[1, 0].tick_params(axis='x', labelrotation = 0)
        axs[1, 0].grid(zorder=0,axis='y')
        
        # Plot the fourth chart (bottom-right - Line chart)
        axs[1, 1].plot(df4[x], df4['vl_intra_task_execution_time_device_func_cpu'], color='C2', linestyle = 'dotted', linewidth=2.5)
        axs[1, 1].plot(df4[x], df4['vl_intra_task_execution_time_device_func_gpu'], color='C2', linestyle = 'solid', linewidth=2.5)
        axs[1, 1].plot(df4[x], df4['vl_communication_time_gpu'], color='C4', linestyle = 'solid', linewidth=2.5)
        axs[1, 1].legend(loc=(-0.000,0.99), frameon=False, labelspacing=0.01, ncol=3, borderpad=0.1)
        axs[1, 1].set_xlabel('Block size MB')  # Add x-axis label
        axs[1, 1].set_yscale('log')
        axs[1, 1].set_ylim([1e-3, 1e4])
        axs[1, 1].tick_params(axis='x', labelrotation = 0)
        axs[1, 1].grid(zorder=0,axis='y')
        
        # Adjust layout to prevent clipping of titles and labels
        plt.tight_layout()

        # Adjust spacing
        plt.subplots_adjust(wspace=0.01, hspace=0.25)
        
        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)


    elif query == 'query_9A.sql':
        
        # Filtering query
        df_filtered = df[
                        (df["ds_resource"] == "MINOTAURO_9_NODES_1_CORE")
                        & (df["ds_parameter_type"].isin(["VAR_GRID_ROW_5","VAR_GRID_ROW_6","VAR_GRID_ROW_7"]))
                        & (df["ds_dataset"] == "S_10GB_1")
                        ]
        

        # Start Plot
        matplotlib.rcParams.update({'font.size': 18})

        # K-means Speedup
        # nr_cluster = 10
        df_filtered_left_top = df_filtered[(df_filtered.nr_cluster==10)]
        # nr_cluster = 100
        df_filtered_center_top = df_filtered[(df_filtered.nr_cluster==100)]
        # nr_cluster = 1000
        df_filtered_right_top = df_filtered[(df_filtered.nr_cluster==1000)]
        
        # K-means Time
        # nr_cluster = 10
        df_filtered_left_bottom = df_filtered[(df_filtered.nr_cluster==10)]
        # nr_cluster = 100
        df_filtered_center_bottom = df_filtered[(df_filtered.nr_cluster==100)]
        # nr_cluster = 1000
        df_filtered_right_bottom = df_filtered[(df_filtered.nr_cluster==1000)]

        left_title = '10 clusters'
        center_title = '100 clusters'
        right_title = '1000 clusters'

        # Speedups
        df1 = df_filtered_left_top[["vl_block_memory_size","vl_block_memory_size_mb","speedup_gpu_intra_task_execution_time_full_func"]].sort_values(by=["vl_block_memory_size"], ascending=[True])
        df2 = df_filtered_center_top[["vl_block_memory_size","vl_block_memory_size_mb","speedup_gpu_intra_task_execution_time_full_func"]].sort_values(by=["vl_block_memory_size"], ascending=[True])
        df3 = df_filtered_right_top[["vl_block_memory_size","vl_block_memory_size_mb","speedup_gpu_intra_task_execution_time_full_func"]].sort_values(by=["vl_block_memory_size"], ascending=[True])

        # Times
        df4 = df_filtered_left_bottom[["vl_block_memory_size","vl_block_memory_size_mb","vl_intra_task_execution_time_device_func_cpu","vl_additional_time_cpu","vl_communication_time_cpu","vl_intra_task_execution_time_device_func_gpu","vl_additional_time_gpu","vl_communication_time_gpu"]].sort_values(by=["vl_block_memory_size"], ascending=[True])
        df5 = df_filtered_center_bottom[["vl_block_memory_size","vl_block_memory_size_mb","vl_intra_task_execution_time_device_func_cpu","vl_additional_time_cpu","vl_communication_time_cpu","vl_intra_task_execution_time_device_func_gpu","vl_additional_time_gpu","vl_communication_time_gpu"]].sort_values(by=["vl_block_memory_size"], ascending=[True])
        df6 = df_filtered_right_bottom[["vl_block_memory_size","vl_block_memory_size_mb","vl_intra_task_execution_time_device_func_cpu","vl_additional_time_cpu","vl_communication_time_cpu","vl_intra_task_execution_time_device_func_gpu","vl_additional_time_gpu","vl_communication_time_gpu"]].sort_values(by=["vl_block_memory_size"], ascending=[True])

        # inserting missing points (for OOM)
        df3 = pd.concat([df3,pd.DataFrame({"vl_block_memory_size":["10000000000"],
                                               "vl_block_memory_size_mb":["10000"],
                                               "vl_intra_task_execution_time_device_func_cpu":[np.nan],
                                               "vl_additional_time_cpu":[np.nan],
                                               "vl_communication_time_cpu":[np.nan],
                                               "vl_intra_task_execution_time_device_func_gpu":[np.nan],
                                               "vl_additional_time_gpu":[np.nan],
                                               "vl_communication_time_gpu":[1000]
                                               })])

        df6 = pd.concat([df6,pd.DataFrame({"vl_block_memory_size":["10000000000"],
                                               "vl_block_memory_size_mb":["10000"],
                                               "vl_intra_task_execution_time_device_func_cpu":[np.nan],
                                               "vl_additional_time_cpu":[np.nan],
                                               "vl_communication_time_cpu":[np.nan],
                                               "vl_intra_task_execution_time_device_func_gpu":[np.nan],
                                               "vl_additional_time_gpu":[np.nan],
                                               "vl_communication_time_gpu":[1000]
                                               })])

        x = 'vl_block_memory_size_mb'
        x_value = "vl_block_memory_size_mb"

        fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharey='row', sharex='col')

        # Plot the first chart (top-left - Bar chart)
        axs[0, 0].bar(df1[x_value],df1['speedup_gpu_intra_task_execution_time_full_func'], color='C0', alpha = 0.5, label='Usr. Code')
        axs[0, 0].legend(loc=(-0.000,0.99), frameon=False, labelspacing=0.01, ncol=3, borderpad=0.1)
        axs[0, 0].set_ylabel('GPU Speedup over CPU')  # Add y-axis label
        axs[0, 0].set_ylim([0, 8])
        axs[0, 0].grid(zorder=0,axis='y')
        axs[0, 0].set_title(left_title, pad=25)

        # Plot the second chart (top-center - Bar chart)
        axs[0, 1].bar(df2[x],df2['speedup_gpu_intra_task_execution_time_full_func'],color='C0', alpha = 0.5)
        axs[0, 1].set_ylim([0, 8])
        axs[0, 1].grid(zorder=0,axis='y')
        axs[0, 1].set_title(center_title, pad=25)

        # Plot the third chart (top-right - Bar chart)
        axs[0, 2].bar(df3[x],df3['speedup_gpu_intra_task_execution_time_full_func'],color='C0', alpha = 0.5)
        axs[0, 2].set_ylim([0, 8])
        axs[0, 2].grid(zorder=0,axis='y')
        axs[0, 2].set_title(right_title, pad=25)

        # Plot the fourth chart (bottom-left - Line chart)
        axs[1, 0].plot(df4[x], df4['vl_intra_task_execution_time_device_func_cpu'], color='C2', linestyle = 'dotted', label='P. Frac. CPU', linewidth=2.5)
        axs[1, 0].plot(df4[x], df4['vl_additional_time_cpu'], color='C8', linestyle = 'dotted', label='S. Frac. GPU', linewidth=2.5)
        axs[1, 0].plot(df4[x], df4['vl_intra_task_execution_time_device_func_gpu'], color='C2', linestyle = 'solid', label='P. Frac. GPU', linewidth=2.5)
        axs[1, 0].plot(df4[x], df4['vl_additional_time_gpu'], color='C8', linestyle = 'solid', label='S. Frac. GPU', linewidth=2.5)
        axs[1, 0].plot(df4[x], df4['vl_communication_time_gpu'], color='C4', linestyle = 'solid', label='CPU-GPU Comm.', linewidth=2.5)
        axs[1, 0].legend(loc=(-0.000,0.99), frameon=False, labelspacing=0.01, ncol=5, borderpad=0.1)
        axs[1, 0].set_xlabel('Block size MB')  # Add x-axis label
        axs[1, 0].set_ylabel('Average Time per Task (s)')  # Add y-axis label
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_ylim([1e-3, 1e3])
        axs[1, 0].tick_params(axis='x', labelrotation = 30)
        axs[1, 0].grid(zorder=0,axis='y')
        
        # Plot the fifth chart (bottom-center - Line chart)
        axs[1, 1].plot(df5[x], df5['vl_intra_task_execution_time_device_func_cpu'], color='C2', linestyle = 'dotted', label='P. Frac. CPU', linewidth=2.5)
        axs[1, 1].plot(df5[x], df5['vl_additional_time_cpu'], color='C8', linestyle = 'dotted', label='S. Frac. GPU', linewidth=2.5)
        axs[1, 1].plot(df5[x], df5['vl_intra_task_execution_time_device_func_gpu'], color='C2', linestyle = 'solid', label='P. Frac. GPU', linewidth=2.5)
        axs[1, 1].plot(df5[x], df5['vl_additional_time_gpu'], color='C8', linestyle = 'solid', label='S. Frac. GPU', linewidth=2.5)
        axs[1, 1].plot(df5[x], df5['vl_communication_time_gpu'], color='C4', linestyle = 'solid', label='CPU-GPU Comm.', linewidth=2.5)
        axs[1, 1].set_xlabel('Block size MB')  # Add x-axis label
        axs[1, 1].set_yscale('log')
        axs[1, 1].set_ylim([1e-3, 1e3])
        axs[1, 1].tick_params(axis='x', labelrotation = 30)
        axs[1, 1].grid(zorder=0,axis='y')

        # Plot the sixth chart (bottom-right - Line chart)
        axs[1, 2].plot(df6[x], df6['vl_intra_task_execution_time_device_func_cpu'], color='C2', linestyle = 'dotted', label='P. Frac. CPU', linewidth=2.5)
        axs[1, 2].plot(df6[x], df6['vl_additional_time_cpu'], color='C8', linestyle = 'dotted', label='S. Frac. GPU', linewidth=2.5)
        axs[1, 2].plot(df6[x], df6['vl_intra_task_execution_time_device_func_gpu'], color='C2', linestyle = 'solid', label='P. Frac. GPU', linewidth=2.5)
        axs[1, 2].plot(df6[x], df6['vl_additional_time_gpu'], color='C8', linestyle = 'solid', label='S. Frac. GPU', linewidth=2.5)
        axs[1, 2].plot(df6[x], df6['vl_communication_time_gpu'], color='C4', linestyle = 'solid', label='CPU-GPU Comm.', linewidth=2.5)
        axs[1, 2].set_xlabel('Block size MB')  # Add x-axis label
        axs[1, 2].set_yscale('log')
        axs[1, 2].set_ylim([1e-3, 1e3])
        axs[1, 2].tick_params(axis='x', labelrotation = 30)
        axs[1, 2].grid(zorder=0,axis='y')

        # Adjust layout to prevent clipping of titles and labels
        plt.tight_layout()

        # Adjust spacing
        plt.subplots_adjust(wspace=0.01, hspace=0.25)

        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)


    elif query == 'query_9B1.sql':
        
        # Filter query
        df_filtered = df[
                        (df["ds_resource"] == "MINOTAURO_9_NODES_1_CORE")
                        & (df["ds_dataset"].isin(["S_2GB_1","S_2GB_3"]))
                        & (df["ds_parameter_type"] == "VAR_GRID_SHAPE_MATMUL_1")
                        ]
        

        # Start Plot
        matplotlib.rcParams.update({'font.size': 18})

        ds_dataset = df_filtered["ds_dataset"].unique()
        ds_dataset = '(' + ', '.join(ds_dataset) + ')'

        x_value = 'vl_concat_block_size_mb_grid_row_x_column_dimension'

        df_filtered_mean = df_filtered.groupby([x_value,'device_skewness'], as_index=False).mean()

        df_filtered_mean.sort_values(by=['vl_grid_row_dimension'], ascending=[False], inplace=True)
        
        df_filtered_mean = df_filtered_mean[[x_value,'device_skewness','vl_total_execution_time','vl_intra_task_execution_time_full_func','vl_intra_task_execution_time_device_func']]
        
        X_axis = np.arange(len(df_filtered_mean[x_value].drop_duplicates()))
        
        fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharey='row', sharex='col')

        # Plot the first chart (top - Bar chart)
        axs[0].bar(X_axis - 0.2,df_filtered_mean[(df_filtered_mean.device_skewness=="CPU NOT SKEWED")]["vl_intra_task_execution_time_full_func"], 0.3, label = "0% Skewness", color='C0', alpha = 0.5, zorder=3)
        axs[0].bar(X_axis + 0.2,df_filtered_mean[(df_filtered_mean.device_skewness=="CPU SKEWED")]["vl_intra_task_execution_time_full_func"], 0.3, label = "50% Skewness", color='red', alpha = 0.5, hatch='oo', zorder=3)
        axs[0].legend(loc=(-0.000,0.99), frameon=False, labelspacing=0.01, ncol=2, borderpad=0.1)
        axs[0].set_ylabel('Usr. Code Time CPU (s)')  # Add y-axis label
        axs[0].grid(zorder=0,axis='y')
        axs[0].set_title('Matmul', pad=20)
        axs[0].set_xticks(X_axis, df_filtered_mean[x_value].drop_duplicates(), rotation=30)

        # Plot the second chart (bottom - Bar chart)
        axs[1].bar(X_axis - 0.2, df_filtered_mean[(df_filtered_mean.device_skewness=="GPU NOT SKEWED")]["vl_intra_task_execution_time_full_func"], 0.3, color='C0', alpha = 0.5, zorder=3)
        axs[1].bar(X_axis + 0.2, df_filtered_mean[(df_filtered_mean.device_skewness=="GPU SKEWED")]["vl_intra_task_execution_time_full_func"], 0.3, color='red', alpha = 0.5, hatch='oo', zorder=3)
        axs[1].set_ylabel('Usr. Code Time GPU (s)')  # Add y-axis label
        axs[1].set_xlabel('Block size MB (Grid Dimension)')
        axs[1].grid(zorder=0,axis='y')
        axs[1].set_xticks(X_axis, df_filtered_mean[x_value].drop_duplicates(), rotation=30)

        fig.autofmt_xdate(rotation=30, ha='right')

        # Adjust spacing
        plt.subplots_adjust(wspace=0.01, hspace=0.25)
        
        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)

    
    elif (query == 'query_9B2.sql'):
        
        # Filter query
        df_filtered = df[
                        (df["ds_resource"] == "MINOTAURO_9_NODES_1_CORE")
                        & (df["ds_dataset"].isin(["S_1GB_1","S_1GB_3"]))
                        & (df["ds_parameter_type"] == "VAR_GRID_ROW_5")
                        ]
        
        matplotlib.rcParams.update({'font.size': 18})

        ds_dataset = df_filtered["ds_dataset"].unique()
        ds_dataset = '(' + ', '.join(ds_dataset) + ')'

        x_value = 'vl_concat_block_size_mb_grid_row_x_column_dimension'

        df_filtered_mean = df_filtered.groupby([x_value,'device_skewness'], as_index=False).mean()

        df_filtered_mean.sort_values(by=['vl_grid_row_dimension'], ascending=[False], inplace=True)
        
        df_filtered_mean = df_filtered_mean[[x_value,'device_skewness','vl_total_execution_time','vl_intra_task_execution_time_full_func','vl_intra_task_execution_time_device_func']]
        
        X_axis = np.arange(len(df_filtered_mean[x_value].drop_duplicates()))
        
        fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharey='row', sharex='col')

        # Plot the first chart (top - Bar chart)
        axs[0].bar(X_axis - 0.2,df_filtered_mean[(df_filtered_mean.device_skewness=="CPU NOT SKEWED")]["vl_intra_task_execution_time_full_func"], 0.3, label = "Dataset 0% Skewed", color='C0', alpha = 0.5, zorder=3)
        axs[0].bar(X_axis + 0.2,df_filtered_mean[(df_filtered_mean.device_skewness=="CPU SKEWED")]["vl_intra_task_execution_time_full_func"], 0.3, label = "Dataset 50% Skewed", color='red', alpha = 0.5, hatch='oo', zorder=3)
        axs[0].set_ylabel('Usr. Code Time CPU (s)')  # Add y-axis label
        axs[0].grid(zorder=0,axis='y')
        axs[0].set_ylim([0, 3])
        axs[0].set_title('K-means', pad=20)
        axs[0].set_xticks(X_axis, df_filtered_mean[x_value].drop_duplicates(), rotation=30)

        # Plot the second chart (bottom - Bar chart)
        axs[1].bar(X_axis - 0.2, df_filtered_mean[(df_filtered_mean.device_skewness=="GPU NOT SKEWED")]["vl_intra_task_execution_time_full_func"], 0.3, color='C0', alpha = 0.5, zorder=3)
        axs[1].bar(X_axis + 0.2, df_filtered_mean[(df_filtered_mean.device_skewness=="GPU SKEWED")]["vl_intra_task_execution_time_full_func"], 0.3, color='red', alpha = 0.5, hatch='oo', zorder=3)
        axs[1].set_ylabel('Usr. Code Time GPU (s)')  # Add y-axis label
        axs[1].set_xlabel('Block size MB (Grid Dimension)')
        axs[1].grid(zorder=0,axis='y')
        axs[1].set_ylim([0,  3])
        axs[1].set_xticks(X_axis, df_filtered_mean[x_value].drop_duplicates(), rotation=30)

        fig.autofmt_xdate(rotation=30, ha='right')

        # Adjust spacing
        plt.subplots_adjust(wspace=0.01, hspace=0.25)
        
        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)
    
    
    elif (query == 'query_10A.sql'):

        # Filter query
        df_filtered = df[
                        (df["ds_resource"] == "MINOTAURO_9_NODES_1_CORE")
                        & (df["ds_parameter_type"].isin(["VAR_GRID_SHAPE_MATMUL_6","VAR_GRID_SHAPE_MATMUL_2","VAR_GRID_SHAPE_MATMUL_1","VAR_GRID_SHAPE_MATMUL_5"])) # FIXED VALUE
                        & (df["ds_dataset"] == "S_8GB_1")
                        & (df["ds_function"] == "MATMUL_FUNC")
                        ]

        matplotlib.rcParams.update({'font.size': 18})

        x = 'vl_concat_block_size_mb_grid_row_x_column_dimension'
        x_value = "vl_concat_block_size_mb_grid_row_x_column_dimension"

        df_filtered_mean = df_filtered.groupby(['ds_device', 'ds_parameter_type', x_value], as_index=False).mean()

        # Matmul Time 
        # local order
        df_filtered_mean_left_top_cpu = df_filtered_mean[(df_filtered_mean.ds_device=='CPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_SHAPE_MATMUL_6')]
        df_filtered_mean_left_top_gpu = df_filtered_mean[(df_filtered_mean.ds_device=='GPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_SHAPE_MATMUL_6')]
        # local data
        df_filtered_mean_right_top_cpu = df_filtered_mean[(df_filtered_mean.ds_device=='CPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_SHAPE_MATMUL_2')]
        df_filtered_mean_right_top_gpu = df_filtered_mean[(df_filtered_mean.ds_device=='GPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_SHAPE_MATMUL_2')]
        
        # Matmul Time
        # shared order
        df_filtered_mean_left_bottom_cpu = df_filtered_mean[(df_filtered_mean.ds_device=='CPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_SHAPE_MATMUL_1')]
        df_filtered_mean_left_bottom_gpu = df_filtered_mean[(df_filtered_mean.ds_device=='GPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_SHAPE_MATMUL_1')]
        # local order
        df_filtered_mean_right_bottom_cpu = df_filtered_mean[(df_filtered_mean.ds_device=='CPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_SHAPE_MATMUL_5')]
        df_filtered_mean_right_bottom_gpu = df_filtered_mean[(df_filtered_mean.ds_device=='GPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_SHAPE_MATMUL_5')]
        
        df_filtered_mean_left_top_cpu.name = 'df_filtered_mean_left_top_cpu'
        df_filtered_mean_left_top_gpu.name = 'df_filtered_mean_left_top_gpu'
        df_filtered_mean_right_top_cpu.name = 'df_filtered_mean_right_top_cpu'
        df_filtered_mean_right_top_gpu.name = 'df_filtered_mean_right_top_gpu'
        df_filtered_mean_left_bottom_cpu.name = 'df_filtered_mean_left_bottom_cpu'
        df_filtered_mean_left_bottom_gpu.name = 'df_filtered_mean_left_bottom_gpu'
        df_filtered_mean_right_bottom_cpu.name = 'df_filtered_mean_right_bottom_cpu'
        df_filtered_mean_right_bottom_gpu.name = 'df_filtered_mean_right_bottom_gpu'

        # Times
        df1_cpu = df_filtered_mean_left_top_cpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df1_gpu = df_filtered_mean_left_top_gpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df2_cpu = df_filtered_mean_right_top_cpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df2_gpu = df_filtered_mean_right_top_gpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df3_cpu = df_filtered_mean_left_bottom_cpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df3_gpu = df_filtered_mean_left_bottom_gpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df4_cpu = df_filtered_mean_right_bottom_cpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df4_gpu = df_filtered_mean_right_bottom_gpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])

        fig, axs = plt.subplots(1, 4, figsize=(24, 4), sharey='row', sharex='col')

      
        # Plot the first chart (top-left - Bar chart)
        axs[0].plot(df1_cpu[x_value],df1_cpu['vl_total_execution_time'], color='C1', linestyle = 'dotted', label='CPU', zorder=3, linewidth=2.5)
        axs[0].plot(df1_gpu[x_value],df1_gpu['vl_total_execution_time'], color='C1', linestyle = 'solid', label='GPU', zorder=3, linewidth=2.5)
        axs[0].set_xlabel('Block size MB (Grid Dimension)')
        axs[0].legend(loc='upper left', frameon=False, labelspacing=0.01, ncol=2, borderpad=0.1)
        axs[0].set_ylabel('P. Tasks Average Time (s)')  # Add y-axis label
        axs[0].set_ylim([0, 2500])
        axs[0].grid(zorder=0,axis='y')
        axs[0].set_title('Local disk, task generation order', pad=10)

        # Plot the second chart (top-right - Bar chart)
        axs[1].plot(df2_cpu[x_value],df2_cpu['vl_total_execution_time'], color='C1', linestyle = 'dotted', zorder=3, linewidth=2.5)
        axs[1].plot(df2_gpu[x_value],df2_gpu['vl_total_execution_time'], color='C1', linestyle = 'solid', zorder=3, linewidth=2.5)
        axs[1].legend(loc='upper left', frameon=False, labelspacing=0.01, ncol=2, borderpad=0.1)
        axs[1].set_ylim([0, 2500])
        axs[1].grid(zorder=0,axis='y')
        axs[1].set_title('Local disk, data locality', pad=10)

        # Plot the third chart (bottom-left - Line chart)
        axs[2].plot(df3_cpu[x_value],df3_cpu['vl_total_execution_time'], color='C1', linestyle = 'dotted', zorder=3, linewidth=2.5)
        axs[2].plot(df3_gpu[x_value],df3_gpu['vl_total_execution_time'], color='C1', linestyle = 'solid', zorder=3, linewidth=2.5)
        axs[2].set_ylim([0, 2500])
        axs[2].tick_params(axis='x', labelrotation = 35)
        axs[2].grid(zorder=0,axis='y')
        axs[2].set_title('Shared disk, task generation order', pad=10)
        
        
        # Plot the fourth chart (bottom-center - Line chart)
        axs[3].plot(df4_cpu[x_value],df4_cpu['vl_total_execution_time'], color='C1', linestyle = 'dotted', zorder=3, linewidth=2.5)
        axs[3].plot(df4_gpu[x_value],df4_gpu['vl_total_execution_time'], color='C1', linestyle = 'solid', zorder=3, linewidth=2.5)
        axs[3].set_ylim([0, 2500])
        axs[3].tick_params(axis='x', labelrotation = 35)
        axs[3].grid(zorder=0,axis='y')
        axs[3].set_title('Shared disk, data locality', pad=10)
        
        # Adjust layout to prevent clipping of titles and labels
        plt.tight_layout()

        fig.autofmt_xdate(rotation=35, ha='right')

        # Adjust spacing
        plt.subplots_adjust(wspace=0.01, hspace=0.28)
        
        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)

    
    elif (query == 'query_10B.sql'):
        
        # Filter query
        df_filtered = df[
                        (df["ds_resource"] == "MINOTAURO_9_NODES_1_CORE")
                        & (df["ds_parameter_type"].isin(["VAR_GRID_ROW_11","VAR_GRID_ROW_8","VAR_GRID_ROW_5","VAR_GRID_ROW_14"])) # FIXED VALUE
                        & (df["ds_dataset"] == "S_10GB_1")
                        ]
        
        matplotlib.rcParams.update({'font.size': 18})

        x = 'vl_concat_block_size_mb_grid_row_x_column_dimension'
        x_value = "vl_concat_block_size_mb_grid_row_x_column_dimension"

        df_filtered_mean = df_filtered.groupby(['ds_device', 'ds_parameter_type', x_value], as_index=False).mean()

        # K-means Time 
        # local order
        df_filtered_mean_left_top_cpu = df_filtered_mean[(df_filtered_mean.ds_device=='CPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_ROW_11')]
        df_filtered_mean_left_top_gpu = df_filtered_mean[(df_filtered_mean.ds_device=='GPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_ROW_11')]
        # local data
        df_filtered_mean_right_top_cpu = df_filtered_mean[(df_filtered_mean.ds_device=='CPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_ROW_8')]
        df_filtered_mean_right_top_gpu = df_filtered_mean[(df_filtered_mean.ds_device=='GPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_ROW_8')]
        
        # K-means Time
        # shared order
        df_filtered_mean_left_bottom_cpu = df_filtered_mean[(df_filtered_mean.ds_device=='CPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_ROW_5')]
        df_filtered_mean_left_bottom_gpu = df_filtered_mean[(df_filtered_mean.ds_device=='GPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_ROW_5')]
        # local order
        df_filtered_mean_right_bottom_cpu = df_filtered_mean[(df_filtered_mean.ds_device=='CPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_ROW_14')]
        df_filtered_mean_right_bottom_gpu = df_filtered_mean[(df_filtered_mean.ds_device=='GPU') & (df_filtered_mean.ds_parameter_type=='VAR_GRID_ROW_14')]
        
        df_filtered_mean_left_top_cpu.name = 'df_filtered_mean_left_top_cpu'
        df_filtered_mean_left_top_gpu.name = 'df_filtered_mean_left_top_gpu'
        df_filtered_mean_right_top_cpu.name = 'df_filtered_mean_right_top_cpu'
        df_filtered_mean_right_top_gpu.name = 'df_filtered_mean_right_top_gpu'
        df_filtered_mean_left_bottom_cpu.name = 'df_filtered_mean_left_bottom_cpu'
        df_filtered_mean_left_bottom_gpu.name = 'df_filtered_mean_left_bottom_gpu'
        df_filtered_mean_right_bottom_cpu.name = 'df_filtered_mean_right_bottom_cpu'
        df_filtered_mean_right_bottom_gpu.name = 'df_filtered_mean_right_bottom_gpu'

        # Times
        df1_cpu = df_filtered_mean_left_top_cpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df1_gpu = df_filtered_mean_left_top_gpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df2_cpu = df_filtered_mean_right_top_cpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df2_gpu = df_filtered_mean_right_top_gpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df3_cpu = df_filtered_mean_left_bottom_cpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df3_gpu = df_filtered_mean_left_bottom_gpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df4_cpu = df_filtered_mean_right_bottom_cpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])
        df4_gpu = df_filtered_mean_right_bottom_gpu.sort_values(by=["vl_block_row_dimension"], ascending=[True])

        fig, axs = plt.subplots(1, 4, figsize=(24, 4), sharey='row', sharex='col')

        # Plot the first chart (top-left - Bar chart)
        axs[0].plot(df1_cpu[x_value],df1_cpu['vl_inter_task_execution_time'], color='C1', linestyle = 'dotted', label='CPU', zorder=3, linewidth=2.5)
        axs[0].plot(df1_gpu[x_value],df1_gpu['vl_inter_task_execution_time'], color='C1', linestyle = 'solid', label='GPU', zorder=3, linewidth=2.5)
        axs[0].set_xlabel('Block size MB (Grid Dimension)')
        axs[0].legend(loc='upper left', frameon=False, labelspacing=0.01, ncol=2, borderpad=0.1)
        axs[0].set_ylabel('P. Tasks Average Time (s)')  # Add y-axis label
        axs[0].set_ylim([0, 265])
        axs[0].grid(zorder=0,axis='y')
        axs[0].set_title('Local disk, task generation order', pad=10)

        # Plot the second chart (top-right - Bar chart)
        axs[1].plot(df2_cpu[x_value],df2_cpu['vl_inter_task_execution_time'], color='C1', linestyle = 'dotted', zorder=3, linewidth=2.5)
        axs[1].plot(df2_gpu[x_value],df2_gpu['vl_inter_task_execution_time'], color='C1', linestyle = 'solid', zorder=3, linewidth=2.5)
        axs[1].legend(loc='upper left', frameon=False, labelspacing=0.01, ncol=2, borderpad=0.1)
        axs[1].set_ylim([0, 265])
        axs[1].grid(zorder=0,axis='y')
        axs[1].set_title('Local disk, data locality', pad=10)

        # Plot the third chart (bottom-left - Line chart)
        axs[2].plot(df3_cpu[x_value],df3_cpu['vl_inter_task_execution_time'], color='C1', linestyle = 'dotted', zorder=3, linewidth=2.5)
        axs[2].plot(df3_gpu[x_value],df3_gpu['vl_inter_task_execution_time'], color='C1', linestyle = 'solid', zorder=3, linewidth=2.5)
        axs[2].set_ylim([0, 265])
        axs[2].tick_params(axis='x', labelrotation = 35)
        axs[2].grid(zorder=0,axis='y')
        axs[2].set_title('Shared disk, task generation order', pad=10)
        
        
        # Plot the fourth chart (bottom-center - Line chart)
        axs[3].plot(df4_cpu[x_value],df4_cpu['vl_inter_task_execution_time'], color='C1', linestyle = 'dotted', zorder=3, linewidth=2.5)
        axs[3].plot(df4_gpu[x_value],df4_gpu['vl_inter_task_execution_time'], color='C1', linestyle = 'solid', zorder=3, linewidth=2.5)
        axs[3].set_ylim([0, 265])
        axs[3].tick_params(axis='x', labelrotation = 35)
        axs[3].grid(zorder=0,axis='y')
        axs[3].set_title('Shared disk, data locality', pad=10)
        
        # Adjust layout to prevent clipping of titles and labels
        plt.tight_layout()

        fig.autofmt_xdate(rotation=35, ha='right')

        # Adjust spacing
        plt.subplots_adjust(wspace=0.01, hspace=0.28)
        
        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)


    elif (query == 'query_11.sql'):
        
        # PRE-PROCESSING (NORMALIZING DATA)
        min_value = df_filtered["Block size"].min()
        max_value = df_filtered["Block size"].max()
        df_filtered["Block size"] = (df_filtered["Block size"] - min_value) / (max_value - min_value)

        min_value = df_filtered["Computational complexity"].min()
        max_value = df_filtered["Computational complexity"].max()
        df_filtered["Computational complexity"] = (df_filtered["Computational complexity"] - min_value) / (max_value - min_value)
        
        min_value = df_filtered["DAG maximum width"].min()
        max_value = df_filtered["DAG maximum width"].max()
        df_filtered["DAG maximum width"] = (df_filtered["DAG maximum width"] - min_value) / (max_value - min_value)
        
        min_value = df_filtered["DAG maximum height"].min()
        max_value = df_filtered["DAG maximum height"].max()
        df_filtered["DAG maximum height"] = (df_filtered["DAG maximum height"] - min_value) / (max_value - min_value)

        min_value = df_filtered["Dataset size"].min()
        max_value = df_filtered["Dataset size"].max()
        df_filtered["Dataset size"] = (df_filtered["Dataset size"] - min_value) / (max_value - min_value)

        # # CORRELATION MATRIX (PEARSON OR SPEARMAN)
        matplotlib.rcParams.update({'font.size': 20})

        chart_width = 30
        chart_height = 20

        # Create a figure with a fixed size
        fig = plt.figure(figsize=(chart_width, chart_height))

        # Define the size and position of the plot area within the chart
        left_margin = 0.22
        bottom_margin = 0.12
        plot_width = 0.6
        plot_height = 0.4

        # Calculate the position of the plot area
        plot_left = left_margin
        plot_bottom = bottom_margin

        # # Create the plot within the defined plot area
        ax = fig.add_axes([plot_left, plot_bottom, plot_width, plot_height])

        ax = plt.gca()
        corrMatrix = df_filtered.corr(method='spearman')
        # CORRELATION MATRIX
        print(corrMatrix)
        sns.heatmap(corrMatrix, annot=True, fmt='.3f', cmap='coolwarm', annot_kws={'size': 16})
        plt.xticks(rotation=30, ha='right')

        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)


    elif (query == 'query_12.sql'):
        
        # Filter query
        df_filtered = df[
                        (df["ds_resource"] == "MINOTAURO_9_NODES_1_CORE")
                        & (df["ds_dataset"] == "S_8GB_1")
                        & (df["ds_parameter_type"] == "VAR_GRID_SHAPE_MATMUL_1")
                        ]

        matplotlib.rcParams.update({'font.size': 18})

        # Matmul FMA Speedup
        df_filtered_left = df_filtered
        # Matmul FMA Time
        df_filtered_right = df_filtered

        # Speedups
        df1 = df_filtered_left[["vl_block_memory_size","vl_block_memory_size_mb","speedup_gpu_intra_task_execution_time_full_func"]].sort_values(by=["vl_block_memory_size"], ascending=[True])
        # Times
        df2 = df_filtered_right[["vl_block_memory_size","vl_block_memory_size_mb","vl_intra_task_execution_time_device_func_cpu","vl_additional_time_cpu","vl_communication_time_cpu","vl_intra_task_execution_time_device_func_gpu","vl_additional_time_gpu","vl_communication_time_gpu"]].sort_values(by=["vl_block_memory_size"], ascending=[True])

        # use the code below to plot chart with missing values
        # from here
        df1 = pd.concat([df1,pd.DataFrame({"vl_block_memory_size":["8192000000"],
                                               "vl_block_memory_size_mb":["8192"],
                                               "speedup_gpu_intra_task_execution_time_full_func":[0]
                                               })])


        x = 'vl_block_memory_size_mb'
        x_value = "vl_block_memory_size_mb"

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Plot the first chart (top-left - Bar chart)
        axs[0].bar(df1[x_value],df1['speedup_gpu_intra_task_execution_time_full_func'], color='C0', alpha = 0.5, label='Usr. Code')
        axs[0].legend(loc='upper left', frameon=False, labelspacing=0.01, ncol=3, borderpad=0.1)
        axs[0].set_ylabel('GPU Speedup over CPU')  # Add y-axis label
        axs[0].set_xlabel('Block size MB')  # Add x-axis label
        axs[0].set_ylim([0, 25])
        axs[0].grid(zorder=0,axis='y')

        # Plot the second chart (top-right - Bar chart)
        axs[1].plot(df2[x],df2['vl_intra_task_execution_time_device_func_cpu'], color='C2', linestyle = 'dotted', label='P. Frac. CPU', zorder=3, linewidth=2.5)
        axs[1].plot(df2[x],df2['vl_intra_task_execution_time_device_func_gpu'], color='C2', linestyle = 'solid', label='P. Frac. GPU', zorder=3, linewidth=2.5)
        axs[1].plot(df2[x],df2['vl_communication_time_gpu'],color='C4', linestyle = 'solid', label='CPU-GPU Comm.', zorder=3, linewidth=2.5)
        axs[1].legend(loc='upper left', frameon=False, labelspacing=0.01, ncol=1, borderpad=0.1)
        axs[1].set_ylabel('Avg. Time per Task (s)')  # Add y-axis label
        axs[1].set_xlabel('Block size MB')  # Add x-axis label
        axs[1].set_ylim([1e-2, 1e4])
        axs[1].set_yscale('log')
        axs[1].grid(zorder=0,axis='y')

        # Adjust layout to prevent clipping of titles and labels
        plt.tight_layout()

        # Adjust spacing
        plt.subplots_adjust(wspace=0.37, hspace=0.2)
        
        # Save plots
        plt.savefig(dst_path_figs+'fig_'+str(query)+'.png',bbox_inches='tight',dpi=100)

    elif (query == 'query_13.sql'):
        
        # Filter query
        df_filtered = df[
                        (df["ds_resource"] == "MINOTAURO_9_NODES_1_CORE")
                        & (df["ds_dataset"].isin(["S_2GB_1","S_2GB_2"]))
                        & (df["ds_parameter_type"] == "VAR_GRID_SHAPE_MATMUL_1")
                        & (df["ds_function"] == "MATMUL_FUNC")
                        ]
        
        matplotlib.rcParams.update({'font.size': 18})

        ds_dataset = df_filtered["ds_dataset"].unique()
        ds_dataset = '(' + ', '.join(ds_dataset) + ')'
        
        # Define the size of the overall chart area in inches
        chart_width = 6.4
        chart_height = 4.8

        # Create a figure with a fixed size
        fig = plt.figure(figsize=(chart_width, chart_height))

        left_margin = 0.15
        bottom_margin = -0.15
        plot_width = 1
        plot_height = 1

        # Calculate the position of the plot area
        plot_left = left_margin
        plot_bottom = bottom_margin

        # Create the plot within the defined plot area
        ax = fig.add_axes([plot_left, plot_bottom, plot_width, plot_height])

        x_value_list = ['vl_concat_block_size_mb_grid_row_x_column_dimension']

        for x_value in x_value_list:

            if x_value == 'vl_concat_grid_row_x_column_dimension_block_size_mb':
                x_value_title = 'Grid Shape (Block Size MB)'
            elif x_value == 'vl_concat_block_size_mb_grid_row_x_column_dimension':
                x_value_title = 'Block Size MB (Grid Shape)'
            
            df_filtered_mean = df_filtered.groupby([x_value,'device_sparsity'], as_index=False).mean()

            df_filtered_mean.sort_values(by=['vl_grid_row_dimension'], ascending=[False], inplace=True)
            
            df_filtered_mean = df_filtered_mean[[x_value,'device_sparsity','vl_intra_task_execution_time_full_func']]

            plt.figure(2)
            X_axis = np.arange(len(df_filtered_mean[x_value].drop_duplicates()))
            plt.bar(X_axis - 0.2, df_filtered_mean[(df_filtered_mean.device_sparsity=="CPU dense")]["vl_intra_task_execution_time_full_func"], 0.3, label = "Dense dataset", color='C0', alpha = 0.25, zorder=3)
            plt.bar(X_axis + 0.2, df_filtered_mean[(df_filtered_mean.device_sparsity=="CPU sparse")]["vl_intra_task_execution_time_full_func"], 0.3, label = "Sparse dataset", color='C0', alpha = 0.25, hatch='oo', zorder=3)
            plt.xticks(X_axis, df_filtered_mean[x_value].drop_duplicates(), rotation=90)
            plt.xlabel(x_value_title)
            plt.ylabel('User Code Exec. Time CPU (s)')
            plt.grid(zorder=0,axis='y')
            plt.figlegend(loc='upper center', ncol=2, frameon=False)
            plt.ylim([0, 250])
            # Save plots
            plt.savefig(dst_path_figs+'fig_'+str(query)+'_CPU.png',bbox_inches='tight',dpi=100)

            plt.figure(3)
            X_axis = np.arange(len(df_filtered_mean[x_value].drop_duplicates()))
            plt.bar(X_axis - 0.2, df_filtered_mean[(df_filtered_mean.device_sparsity=="GPU dense")]["vl_intra_task_execution_time_full_func"], 0.3, label = "Dense dataset", color='C0', alpha = 0.25, zorder=3)
            plt.bar(X_axis + 0.2, df_filtered_mean[(df_filtered_mean.device_sparsity=="GPU sparse")]["vl_intra_task_execution_time_full_func"], 0.3, label = "Sparse dataset", color='C0', alpha = 0.25, hatch='oo', zorder=3)
            plt.xticks(X_axis, df_filtered_mean[x_value].drop_duplicates(), rotation=90)
            plt.xlabel(x_value_title)
            plt.ylabel('User Code Exec. Time GPU (s)')
            plt.grid(zorder=0,axis='y')
            plt.figlegend(loc='upper center', ncol=2, frameon=False)
            plt.ylim([0, 11])
            # Save plots
            plt.savefig(dst_path_figs+'fig_'+str(query)+'_GPU.png',bbox_inches='tight',dpi=100)
    
    else:
        print('error: invalid query')



def setup_database():

    path_schemas = '/raw_data'

    schemas = get_folder_names(path_schemas)

    # Setup each schema
    for schema in schemas:

        # Open connection to the database
        cur, conn = open_connection(schema)

        # Set database schemas
        sql_create_schema = "CREATE SCHEMA %s;"
        cur.execute(sql_create_schema, (schema))
        # Create tables
        sql_create_table =  "CALL CREATE_TABLES();"
        cur.execute(sql_create_table)
        # Load tables
        sql_load_table = "CALL LOAD_TABLES(%s);"
        cur.execute(sql_load_table,(schema))

        # Close connection to the database
        close_connection(cur, conn)


def get_folder_names(path):
    folder_names = []
    
    # Check if the path exists
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return folder_names

    # Get the list of items in the path
    items = os.listdir(path)

    # Filter out folders from the list
    folder_names = [item for item in items if (os.path.isdir(os.path.join(path, item)) and item != 'paraver')]

    return folder_names


# def get_file_names(folder_path):
#     file_names = []

#     # Check if the folder path exists
#     if not os.path.exists(folder_path):
#         print(f"Folder path '{folder_path}' does not exist.")
#         return file_names

#     # Get the list of items in the folder
#     items = os.listdir(folder_path)

#     # Filter out files from the list
#     file_names = [item for item in items if os.path.isfile(os.path.join(folder_path, item))]

#     return file_names


def get_files_with_prefix(path, prefix):
    file_names = []

    # Check if the path exists
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return file_names

    # Get the list of items in the path
    items = os.listdir(path)

    # Filter out files with the specified prefix
    file_names = [item for item in items if os.path.isfile(os.path.join(path, item)) and item.startswith(prefix) and item != 'query_8A.sql' and item != 'query_8B.sql']

    return file_names

def read_sql_file(file_path):
    try:
        with open(file_path, 'r') as sql_file:
            sql_content = sql_file.read()
            return sql_content
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Function that takes in a PostgreSQL query and outputs a pandas dataframe
def get_df_from_query(sql_query, conn):
    df = pd.read_sql_query(sql_query, conn)
    return df

if __name__ == "__main__":
    main()