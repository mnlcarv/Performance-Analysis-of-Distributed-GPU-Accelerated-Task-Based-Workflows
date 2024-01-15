import dislib as ds

def generate_dataset(vl_dataset_row_dimension, vl_dataset_column_dimension, vl_block_row_dimension, vl_block_column_dimension, vl_grid_row_dimension, vl_grid_column_dimension, nr_random_state, vl_data_skewness, ds_algorithm):

    # dataset structure not belonging to dislib (generate dataset as a blocked numpy array for matmul fma)
    if (ds_algorithm == "MATMUL_FMA"):

        # Generate the dataset in a distributed manner
        # i.e: avoid having the master a whole matrix
        num_blocks = vl_grid_row_dimension
        elems_per_block = vl_block_row_dimension

        A = []
        for i in range(num_blocks):
            for l in [A]:
                l.append([])
            # Keep track of blockId to initialize with different random seeds
            bid = 0
            for j in range(num_blocks):
                for ix, l in enumerate([A]):
                    l[-1].append(ds.generate_block(elems_per_block,
                                                num_blocks,
                                                random_state=nr_random_state,
                                                bid=bid))
                    bid += 1
        return A
    
    # generate dataset as ds-array for dislib algorithms 
    else:

        return ds.random_array((vl_dataset_row_dimension, vl_dataset_column_dimension), (vl_block_row_dimension, vl_block_column_dimension), random_state=nr_random_state, data_skewness=vl_data_skewness)