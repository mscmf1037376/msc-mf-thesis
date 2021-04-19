import unittest
import json
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import munch

from wealth_optimizer.data_slicing import slice_dataset
from wealth_optimizer.common_logger import logger
from wealth_optimizer.data_loader import ExcelDataReader
from wealth_optimizer.executors import ModelExecutor
from wealth_optimizer.models import LongOnlyNoLeverageOneLayerNetwork

from wealth_optimizer.simulated_bm import geometric_brownian_motion_generator, generate_simulated_price_paths


class TestExecutor(unittest.TestCase):

    def test_executor_with_test_config(self):
        config = munch.munchify({
            "use_features": False,
            "train_length": 200,
            "validation_length": 25,
            "test_length": 60,
            "num_epochs": 10,
            "num_train_paths": 10,
            "num_validation_paths": 0,
            "num_test_paths": 10000,
            "horizon": 52,
            "target_wealth": 1.5,
            "initial_wealth": 1.0,
            "expected_block_size": 1,
            "lr": 3e-1,
            "loss_function": "forsyth_li_regularized_loss"
        })
        for k, v in config.__dict__.items():
            logger.info('{}: {}'.format(k, v))

        n_rows = 1000
        n_cols = 2

        bm1 = geometric_brownian_motion_generator(drift=0.020, volatility=0.10)
        bm2 = geometric_brownian_motion_generator(drift=0.40, volatility=0.20)
        bm3 = geometric_brownian_motion_generator(drift=0.40, volatility=0.10)

        spots = np.array([1.0] * 3)
        maturity = 5.0
        num_steps = 5 * 52
        num_paths = 1
        correlation_matrix = np.eye(3)
        # correlation_matrix = np.array([[1.0, 0], [0, 1.0]])
        uncorrelated_gbms = [bm1, bm2, bm3]
        num_assets = len(uncorrelated_gbms)
        # price_paths = generate_simulated_price_paths(np.array([1.0, 1.0]), uncorrelated_gbms, 1.0,
        #                                250,
        #                                1, correlation=np.array([[1.0, 0.4], [0.4, 1.0]]))

        # print(price_paths)
        # print(price_paths.shape)

        asset_paths = generate_simulated_price_paths(spots, uncorrelated_gbms, maturity, num_steps, num_paths,
                                                     correlation_matrix)

        print(asset_paths)

        current_path = asset_paths[:, 0, :]
        df = pd.DataFrame(current_path.transpose())
        df.columns = ['asset_' + str(i + 1) for i in range(num_assets)]
        df.plot()

        plt.show()
        full_data = (df.diff() / df).dropna(inplace=False)
        full_data.index = np.arange(maturity / num_steps, maturity, maturity / num_steps)
        full_data.columns = ['asset_' + str(i + 1) for i in range(num_assets)]
        assert isinstance(full_data, pd.DataFrame)
        print(full_data)
        # input('stop')

        # full_data = pd.DataFrame(np.random.lognormal(mean=0.0, sigma=0.01, size=(n_rows, n_cols)),
        #                          columns=['asset_' + str(x + 1) for x in range(n_cols)])

        full_data.plot()
        plt.show()
        # input('Press Enter')

        full_data = full_data.diff()
        full_data.dropna(inplace=True)

        print(full_data)
        print('======================================================================================================')

        training_data, validation_data, test_data = slice_dataset(full_data, config.test_length,
                                                                  config.validation_length,
                                                                  overlap_validation_training=True)
        training_data = training_data.tail(config.train_length)
        logger.info('Full data samples: {}'.format(len(full_data)))
        logger.info('Training data samples: {}'.format(len(training_data)))
        logger.info('Validation data samples: {}'.format(len(validation_data)))
        logger.info('Test data samples: {}'.format(len(test_data)))

        print("training set index:", training_data.index)
        print("validation set index:", validation_data.index)
        print("test set index:", test_data.index)

        # create executor and run
        executor = ModelExecutor(training_data, validation_data, test_data,
                                 model=LongOnlyNoLeverageOneLayerNetwork,
                                 parameters=config)

        executor.execute()
