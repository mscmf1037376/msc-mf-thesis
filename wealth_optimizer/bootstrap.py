import numpy as np
import pandas as pd


def generate_stationary_bootstrap_path(data: pd.DataFrame, expected_block_size: int, n_req: int):
    bootstrap_samples = []
    data_length = len(data)

    while True:

        # choose random starting index in [0,...,N-1], N-1 is the index of the last historical sample
        index = np.random.choice(data_length)
        # actual blocksize follows a shifted geometric distribution with expected value of exp block size
        blocksize = np.random.geometric(1/expected_block_size)
        for i in range(blocksize):
            # if the chosen block exceeds the range of the historical data array, do a circular bootstrap
            if index + i >= data_length:
                bootstrap_samples.append(data.iloc[index + i - data_length])
            else:
                bootstrap_samples.append(data.iloc[index + i])

            if len(bootstrap_samples) == n_req:

                output_array = np.array(bootstrap_samples).reshape(n_req, len(data.columns))
                # assert output_array.shape[0] == n_req
                # assert output_array.shape[1] == len(data.columns)
                return output_array


def generate_stationary_bootstrap_path_fixed_block_size(data: pd.DataFrame, block_size: int, n_req: int):
    bootstrap_samples = []
    data_length = len(data)

    while True:

        # choose random starting index in [0,...,N-1], N-1 is the index of the last historical sample
        index = np.random.choice(data_length)
        for i in range(block_size):
            # if the chosen block exceeds the range of the historical data array, do a circular bootstrap
            if index + i >= data_length:
                bootstrap_samples.append(data.iloc[index + i - data_length])
            else:
                bootstrap_samples.append(data.iloc[index + i])

            if len(bootstrap_samples) == n_req:

                output_array = np.array(bootstrap_samples).reshape(n_req, len(data.columns))
                # assert output_array.shape[0] == n_req
                # assert output_array.shape[1] == len(data.columns)
                return output_array