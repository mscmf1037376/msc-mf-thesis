from abc import ABC, abstractmethod

import datetime as dt
from typing import Type

import munch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from wealth_optimizer.bootstrap import generate_stationary_bootstrap_path
from wealth_optimizer.common_logger import logger
from wealth_optimizer.loss_functions import LossFunctionSelector
from wealth_optimizer.models import BaseControlModel
from wealth_optimizer.performance_stats import compute_wealth_distribution_statistics
from wealth_optimizer.data_slicing import slice_dataset
from wealth_optimizer.simulated_bm import generate_simulated_price_paths, geometric_brownian_motion_generator


def process_train_path(control_model: BaseControlModel,
                       path: np.ndarray,
                       num_features: int,
                       num_benchmarks: int = 0,
                       verbose: bool = False) -> torch.Tensor:
    """
    Computes the path returns and controls along a training returns path.
    :param control_model: a BaseControlModel instance
    :param path: current training path of returns (numpy array where each column is an asset or indicator)
    :param num_features: number of features (columns in the path array to use as indicators)
    :param num_benchmarks: number of benchmark columns in the data
    :param verbose: if True, logs the path wealth and asset holdings over the path length
    :return: tensor of wealth along the path
    """
    path_states = control_model.compute_path_states(path, num_features=num_features, num_benchmarks=num_benchmarks,
                                                    verbose=False)
    # create the path controls tensor from precomputed values
    path_controls = control_model.model(torch.tensor(np.array(path_states),
                                                     dtype=torch.float32).to(control_model.device))

    # create path returns tensor from market data
    # the returns are computed from the columns in the data that are NOT features (economic indicators or similar)
    # hence we need to select the columns starting from the num_features-th column to the last column
    path_returns = torch.tensor(path[:, (num_features + num_benchmarks):], dtype=torch.float32).to(control_model.device)

    # compute final wealth for path by multiplying all the returns
    # Starting with wealth W at the beginning of a period, and with return R over that period, the wealth at the
    # end of the period is (1 + R)W. We therefore multiply the simple returns over all periods to get the final wealth

    # We have:
    # W_final = (1+R1)(1+R2)....(1+Rn)W
    # where R1, R2, ..., Rn are the returns over n periods, W is the initial wealth, W_final is the final wealth
    final_path_wealths = torch.prod(
        1 + torch.diag(torch.matmul(path_returns, torch.transpose(path_controls, 0, 1))))

    if verbose:
        logger.info('[process_train_path] w = {}, holdings: \n {}'.format(final_path_wealths,
                                                                          path_controls.detach().numpy()))

    return final_path_wealths
    # # print('path: {}, w = {}, holdings: \n {}'.format(l, W, C.detach().numpy()))
    # path_wealth_list.append(final_path_wealths)


class BaseModelExecutor(ABC):

    @abstractmethod
    def validate_executor_state(self, *args, **kwargs):
        """
        Meant to be called after the executor is initialized
        """
        raise NotImplementedError

    @abstractmethod
    def generate_simulated_datasets(self, *args, **kwargs):
        """
        Generates training, validation and test paths
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the model and produce results
        """
        raise NotImplementedError


class ModelExecutor(BaseModelExecutor):
    """
    A class that executes a given control model over a dataset.
    It's expected to have a training, validation and test dataset. The executor generates
    bootstrapped return paths and uses them to train the underlying NN-based control model.
    The performance of the model can be evaluated on a validation set during training and training can be stopped
    based on different criteria in order to prevent overfitting. The model performance is then evaluated on
    bootstrapped paths generated from the test set, as well as the real historical returns (which is the single
    realized returns path).

    Preferably, the test data is completely unseen (based on market
    price returns that have occurred chronologically after returns and features/indicators the training data), so
    that there is no look-ahead bias tha could inflate the performance of the algorithm.
    """

    def __init__(self, data: pd.DataFrame,
                 model: Type[BaseControlModel],
                 parameters: munch.Munch):
        """
        Initialize model executor
        :param data: asset returns and features
        :param model: control model
        :param parameters: configuration parameters
        """

        # data related work
        self.training_data, self.validation_data, self.test_data = self.__process_input_data(data, parameters)
        self.__extract_features_and_assets()

        # Two of the inputs of the network must be the time to go and the current wealth. Other features are optional
        self.num_inputs = 2 + self.num_features

        # The number of outputs is equal to the number of assets
        self.num_outputs = len(self.assets)

        self.parameters = parameters
        # number of hidden units specified in the experiment config
        self.num_hidden = self.parameters.num_hidden_units

        # loss function from the experiment config
        self.loss_function, self.requires_target_wealth = LossFunctionSelector().get_loss_function(
            self.parameters.loss_function)

        self.loss_function_parameters = {}
        required_loss_parameters = LossFunctionSelector().get_required_parameters(self.parameters.loss_function)
        if required_loss_parameters:
            for rp in required_loss_parameters:
                self.loss_function_parameters[rp] = self.parameters.loss_function_parameters.get(rp)

        self.model_leverage = self.parameters.get('leverage', 1.0)

        model_config = dict(num_inputs=self.num_inputs,
                            num_outputs=self.num_outputs,
                            num_hidden=self.num_hidden,
                            initial_wealth=self.parameters.initial_wealth,
                            learning_rate=self.parameters.initial_learning_rate,
                            leverage=self.model_leverage,
                            device='cpu')

        self.control_model = model(model_config, self.training_data)
        self.epoch_data = []

        self.validate_executor_state()

    def validate_executor_state(self):
        """
        Do some validation on the input data and the configuration parameters
        Raises AssertionError or ValueError
        """
        assert isinstance(self.training_data, pd.DataFrame)
        assert isinstance(self.validation_data, pd.DataFrame)
        assert isinstance(self.test_data, pd.DataFrame)

        assert (self.training_data.columns == self.validation_data.columns).all()
        assert (self.validation_data.columns == self.test_data.columns).all()
        assert self.parameters.initial_learning_rate, 'missing learning rate'
        assert self.parameters.expected_block_size_train, 'missing expected block size'
        assert self.parameters.expected_block_size_test, 'missing expected block size'

    @staticmethod
    def __process_input_data(data: pd.DataFrame, parameters: munch.Munch):
        training_data, validation_data, test_data = slice_dataset(data, parameters.test_length,
                                                                  parameters.validation_length,
                                                                  parameters.overlap_validation_training)

        training_data = training_data.tail(parameters.train_length)
        print('Training data')
        print(training_data)
        print('\n----------------\nValidation data')
        print(validation_data)
        print('\n----------------\nTest data')
        print(test_data, '\n')
        logger.info('Full data samples: {}'.format(len(data)))
        logger.info('Training data samples: {}'.format(len(training_data)))
        logger.info('Validation data samples: {}'.format(len(validation_data)))
        logger.info('Test data samples: {}'.format(len(test_data)))
        return training_data, validation_data, test_data

    def __extract_features_and_assets(self):
        """
        Helper function to sort the columns into features and tradeable assets
        Assets are used for computing returns and wealth along price paths
        Features are purely used as inputs to the model to attempt to improve performance, and are not investable assets
        """

        self.assets = [x for x in list(self.training_data.columns) if 'asset_' in x]
        self.features = [x for x in list(self.training_data.columns) if 'feature_' in x]
        self.benchmarks = [x for x in list(self.training_data.columns) if 'benchmark_' in x]
        self.use_benchmark = len(self.benchmarks) > 0
        self.num_features = len(self.features)
        self.num_benchmarks = len(self.benchmarks)

        if self.num_benchmarks > 1:
            raise ValueError('More than 1 benchmark not supported')

    def generate_simulated_datasets(self, num_train_paths, num_validation_paths, num_test_paths, horizon):

        train_paths = []
        validation_paths = []
        test_paths = []

        for _ in tqdm(range(num_train_paths), desc='[ModelExecutor] generating training paths'):
            _path = generate_stationary_bootstrap_path(self.training_data,
                                                       expected_block_size=self.parameters.expected_block_size_train,
                                                       n_req=horizon)
            train_paths.append(_path)

        for _ in tqdm(range(num_validation_paths), desc='[ModelExecutor] generating validation paths'):
            _path = generate_stationary_bootstrap_path(self.validation_data,
                                                       expected_block_size=self.parameters.expected_block_size_train,
                                                       n_req=horizon)
            validation_paths.append(_path)

        for _ in tqdm(range(num_test_paths), desc='[ModelExecutor] generating new test paths'):
            _path = generate_stationary_bootstrap_path(self.test_data,
                                                       expected_block_size=self.parameters.expected_block_size_test,
                                                       n_req=horizon)
            test_paths.append(_path)

        return train_paths, validation_paths, test_paths

    def execute(self):

        # parameters
        num_epochs = self.parameters.max_num_epochs
        num_train_paths = self.parameters.num_train_paths
        num_validation_paths = self.parameters.num_validation_paths
        num_test_paths = self.parameters.num_test_paths
        horizon = self.parameters.horizon  # investment horizon

        if self.requires_target_wealth:
            target_wealth = self.parameters.target_wealth
        else:
            target_wealth = None
        initial_wealth = self.parameters.initial_wealth
        assert isinstance(self.control_model, BaseControlModel)

        ################################################################################################################
        # generate training, validation and test paths
        train_paths, validation_paths, test_paths = self.generate_simulated_datasets(num_train_paths,
                                                                                     num_validation_paths,
                                                                                     num_test_paths,
                                                                                     horizon)

        ################################################################################################################
        # set up target wealth vector and some variables needed for the training loop
        ################################################################################################################
        if self.requires_target_wealth:
            target_wealths = torch.from_numpy(target_wealth * np.ones(num_train_paths)).float().to(
                self.control_model.device)
        else:
            target_wealths = None
        if num_validation_paths > 0:
            if self.requires_target_wealth:
                target_wealths_validation = torch.from_numpy(target_wealth * np.ones(num_validation_paths)).float().to(
                    self.control_model.device)
            else:
                target_wealths_validation = None
        else:
            target_wealths_validation = None

        self.control_model.model.to(self.control_model.device)
        logger.info('[ModelExecutor] selected device: {}'.format(self.control_model.device))
        logger.info('[ModelExecutor] selected loss function: {}'.format(self.loss_function.__name__))
        logger.info('[ModelExecutor] training started')
        self.control_model.model.train()
        stop_training = False
        prev_training_loss = None
        prev_validation_loss = None
        consecutive_validation_loss_increases = []

        ################################################################################################################
        # training loop
        ################################################################################################################
        for k in range(num_epochs):

            if stop_training:
                # If this flag was set to True during a training epoch, the training stops
                # This can occur if the training loss or validation loss start increasing and
                # the optimization cannot recover (i.e. the increase in loss was not temporary jitter)
                break

            logger.info('[ModelExecutor] epoch {} started'.format(k + 1))
            path_wealth_list = []
            epoch_start = dt.datetime.now()
            self.control_model.model.train()

            for l, path in enumerate(train_paths):
                final_path_wealths = process_train_path(self.control_model, path, self.num_features,
                                                        self.num_benchmarks, verbose=False)
                path_wealth_list.append(final_path_wealths)

            path_wealth_tensor = torch.stack(path_wealth_list).float()

            if target_wealths is not None:
                if self.loss_function_parameters:
                    loss = self.loss_function(path_wealth_tensor, target_wealths, **self.loss_function_parameters)
                else:
                    loss = self.loss_function(path_wealth_tensor, target_wealths)
            else:
                if self.loss_function_parameters:
                    loss = self.loss_function(path_wealth_tensor, None, **self.loss_function_parameters)
                else:
                    loss = self.loss_function(path_wealth_tensor, None)
            detached_wealth_array = path_wealth_tensor.detach().cpu().numpy()

            training_wealths_stats = compute_wealth_distribution_statistics(detached_wealth_array,
                                                                            initial_wealth=initial_wealth,
                                                                            target_wealth=target_wealth)
            average_epoch_wealth = training_wealths_stats.mean
            median_epoch_wealth = training_wealths_stats.median
            min_epoch_wealth = training_wealths_stats.min
            max_epoch_wealth = training_wealths_stats.max
            fraction_above_target = training_wealths_stats.above_target_wealth
            fraction_above_initial = training_wealths_stats.above_initial_wealth

            # compute gradients and zero the gradients
            loss.backward()
            self.control_model.optimizer.step()
            self.control_model.optimizer.zero_grad()

            epoch_end = dt.datetime.now()
            epoch_time = (epoch_end - epoch_start).total_seconds()

            # check performance on validation set
            validation_wealths = []
            validation_controls = []
            if num_validation_paths > 0:
                for validation_path in tqdm(validation_paths, desc='[ModelExecutor] evaluating validation performance'):
                    w, w_path, v_control, _ = self.control_model.compute_path_wealth_and_controls(validation_path,
                                                                                                  self.num_features,
                                                                                                  self.num_benchmarks)

                    validation_wealths.append(w)
                    validation_controls.append(v_control)

                validation_paths_stats = compute_wealth_distribution_statistics(validation_wealths,
                                                                                initial_wealth=initial_wealth,
                                                                                target_wealth=target_wealth)
            else:
                validation_paths_stats = None
            validation_path_wealth_tensor = torch.tensor(validation_wealths).float()
            if num_validation_paths > 0:
                if self.loss_function_parameters:
                    validation_loss = self.loss_function(validation_path_wealth_tensor, target_wealths_validation,
                                                         **self.loss_function_parameters)
                else:
                    validation_loss = self.loss_function(validation_path_wealth_tensor, target_wealths_validation)
            else:
                validation_loss = None

            # print epoch stats
            if validation_loss:
                logger.info(
                    '[ModelExecutor] epoch: {}, Train Loss: {:.8f}, Validation Loss: {:.8f}'.format(
                        k + 1,
                        loss,
                        validation_loss))
            else:
                logger.info(
                    '[ModelExecutor] epoch: {}, Train Loss: {:.8f}'.format(k + 1, loss))

            logger.info('--->  training stats -> avg: {:.4f}, median: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
                average_epoch_wealth,
                median_epoch_wealth,
                min_epoch_wealth,
                max_epoch_wealth))

            logger.info(
                '--->  Prob(W_T > W_*: {}, Prob (W_T > 1.0): {} || compute time: {:.2f}'.format(fraction_above_target,
                                                                                                fraction_above_initial,
                                                                                                epoch_time))

            if validation_paths_stats:
                logger.info('---> validation stats -> avg: {:.4f}, median: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
                    validation_paths_stats.mean,
                    validation_paths_stats.median,
                    validation_paths_stats.min,
                    validation_paths_stats.max))

            epoch_info = {
                'epoch': k + 1,
                'training_loss': loss.detach().numpy(),
                'validation_loss': validation_loss,
                'fraction_above_target': fraction_above_target,
                'fraction_above_initial': fraction_above_initial
            }

            self.epoch_data.append(epoch_info)

            if prev_training_loss:
                logger.info('[ModelExecutor] prev_training_loss: {}'.format(prev_training_loss))
                if loss.detach().numpy() >= prev_training_loss:
                    self.control_model.update_learning_rate()
            prev_training_loss = loss.detach().numpy()

            # early stopping is activated after min_num_epochs and stops training if
            # the validation loss starts to increase
            if k > self.parameters.min_num_epochs:
                if prev_validation_loss:
                    if validation_loss > prev_validation_loss:
                        consecutive_validation_loss_increases.append(k)

                    if len(consecutive_validation_loss_increases) > 3:
                        stop_training = True

            prev_validation_loss = validation_loss

        ################################################################################################################
        # end training loop
        ################################################################################################################

        logger.info('[ModelExecutor] training completed\n')
        logger.info('[ModelExecutor] processing results')

        test_paths_final_wealths = []
        test_paths_wealth_trajectories = []  # evolution of the wealth for each path
        test_paths_controls = []
        test_paths_benchmark_wealths = []

        ################################################################################################################
        # generate results
        ################################################################################################################

        # recompute path wealths along training paths ###########################################################
        train_paths_controls = []
        train_paths_wealth_trajectories = []
        train_paths_final_wealths = []

        validation_paths_controls = []
        validation_paths_wealth_trajectories = []
        validation_paths_final_wealths = []

        logger.info('[ModelExecutor] evaluating model on train paths')
        # use the final trained model to compute the wealths and controls along the training and validation paths
        for train_path in train_paths:
            w, w_path, w_controls, _ = self.control_model.compute_path_wealth_and_controls(train_path,
                                                                                           self.num_features,
                                                                                           self.num_benchmarks)
            train_paths_final_wealths.append(w)
            train_paths_wealth_trajectories.append(w_path)
            train_paths_controls.append(w_controls)

        logger.info('[ModelExecutor] evaluating model on validation paths')
        for validation_path in validation_paths:
            w, w_path, w_controls, _ = self.control_model.compute_path_wealth_and_controls(validation_path,
                                                                                           self.num_features,
                                                                                           self.num_benchmarks)
            validation_paths_final_wealths.append(w)
            validation_paths_wealth_trajectories.append(w_path)
            validation_paths_controls.append(w_controls)

        logger.info('[ModelExecutor] evaluating model on test paths')
        for test_path in test_paths:
            w, w_path, w_controls, benchmark_wealth = self.control_model.compute_path_wealth_and_controls(test_path,
                                                                                                          self.num_features,
                                                                                                          self.num_benchmarks)
            test_paths_final_wealths.append(w)
            test_paths_benchmark_wealths.append(benchmark_wealth)
            # print(w, benchmark_wealth)
            # test_paths_strategy_excess_wealths_over_benchmark.append(w - benchmark_wealth)
            test_paths_wealth_trajectories.append(w_path)
            test_paths_controls.append(w_controls)

        logger.info('[ModelExecutor] evaluating model on historical test path')
        # actual test data - historical backtest
        historical_test_path = self.test_data.to_numpy()
        w_backtest, w_path_backtest, w_controls_backtest, _ = self.control_model.compute_path_wealth_and_controls(
            historical_test_path,
            self.num_features,
            self.num_benchmarks)

        ################################################################################################################

        logger.info('[ModelExecutor] computing wealth distribution statistics')
        train_wealths_stats = compute_wealth_distribution_statistics(np.array(train_paths_final_wealths),
                                                                     initial_wealth=initial_wealth,
                                                                     target_wealth=target_wealth)

        if validation_paths_final_wealths:
            validation_wealths_stats = compute_wealth_distribution_statistics(np.array(validation_paths_final_wealths),
                                                                          initial_wealth=initial_wealth,
                                                                          target_wealth=target_wealth)
        else:
            validation_wealths_stats = None
        test_wealths_stats = compute_wealth_distribution_statistics(np.array(test_paths_final_wealths),
                                                                    initial_wealth=initial_wealth,
                                                                    target_wealth=target_wealth)

        epoch_training_losses = [edata['training_loss'] for edata in self.epoch_data]
        epoch_validation_losses = [edata['validation_loss'] for edata in self.epoch_data]
        epochs = [i for i, _ in enumerate(self.epoch_data)]

        logger.info('[ModelExecutor] constructing result object')

        result_set = {
            'config': {
                'assets': self.assets
            },
            'training': {
                'return_paths': train_paths,
                'controls': train_paths_controls,
                'final_wealth': train_paths_final_wealths,
                'wealth_trajectories': train_paths_wealth_trajectories,
                'stats': train_wealths_stats,
                'loss': epoch_training_losses,
                'epochs': epochs
            },

            'validation': {
                'return_paths': validation_paths,
                'controls': validation_paths_controls,
                'final_wealth': validation_paths_final_wealths,
                'wealth_trajectories': validation_paths_wealth_trajectories,
                'stats': validation_wealths_stats,
                'loss': epoch_validation_losses,
                'epochs': epochs
            },

            'test': {
                'return_paths': test_paths,
                'controls': test_paths_controls,
                'final_wealth': test_paths_final_wealths,
                'wealth_trajectories': test_paths_wealth_trajectories,
                'stats': test_wealths_stats,
                'backtest': {
                    'final_wealth': w_backtest,
                    'wealth_trajectories': w_path_backtest,
                    'controls': w_controls_backtest
                }
            }

        }

        result_object = munch.Munch.fromDict(result_set)
        logger.info('[ModelExecutor] done ---------------------')

        return result_object


class SimulatedGBMModelExecutor(BaseModelExecutor):
    def __init__(self,
                 model: Type[BaseControlModel],
                 parameters: munch.Munch):
        # data related work
        bm1 = geometric_brownian_motion_generator(drift=0.06, volatility=0.30)
        bm2 = geometric_brownian_motion_generator(drift=0.01, volatility=0.0)
        # bm3 = geometric_brownian_motion_generator(drift=0.10, volatility=0.30)

        self.spots = np.array([1.0] * 2)
        self.assets = ['asset_1', 'asset_2']
        self.maturity = 1.0
        self.num_steps = parameters.horizon
        # self.num_paths = 1
        self.correlation_matrix = np.eye(2)
        # correlation_matrix = np.array([[1.0, 0], [0, 1.0]])
        self.uncorrelated_gbms = [bm1, bm2]
        self.num_assets = len(self.uncorrelated_gbms)

        dummy_paths = generate_simulated_price_paths(self.spots, self.uncorrelated_gbms,
                                                     self.maturity,
                                                     self.num_steps, 1,
                                                     self.correlation_matrix)

        # asset_paths = generate_simulated_price_paths(self.spots, self.uncorrelated_gbms,
        #                                              self.maturity,
        #                                              self.num_steps, self.num_paths,
        #                                              self.correlation_matrix)
        print(dummy_paths)

        # print(asset_paths)
        # f, subPlots = plt.subplots(2, sharex=True)
        # for i in range(len(uncorrelated_gbms)):
        #     for j in range(num_paths):
        #         subPlots[i].plot(asset_paths[i, j, :])
        # plt.show()
        #
        # for j in range(1):
        current_path = dummy_paths[:, 0, :]
        df = pd.DataFrame(current_path.transpose())
        df.columns = ['asset_' + str(i + 1) for i in range(self.num_assets)]
        # df.plot()
        returns = (df.diff() / df).dropna(inplace=False)
        returns.index = np.arange(self.maturity / self.num_steps, self.maturity, self.maturity / self.num_steps)
        returns.columns = ['asset_' + str(i + 1) for i in range(self.num_assets)]
        assert isinstance(returns, pd.DataFrame)

        self.num_inputs = 2
        self.num_features = 0
        self.num_benchmarks = 0

        # The number of outputs is equal to the number of assets
        self.num_outputs = self.num_assets

        self.parameters = parameters
        # number of hidden units specified in the experiment config
        self.num_hidden = self.parameters.num_hidden_units

        # loss function from the experiment config
        self.loss_function, self.requires_target_wealth = LossFunctionSelector().get_loss_function(
            self.parameters.loss_function)

        self.loss_function_parameters = {}
        required_loss_parameters = LossFunctionSelector().get_required_parameters(self.parameters.loss_function)
        if required_loss_parameters:
            for rp in required_loss_parameters:
                self.loss_function_parameters[rp] = self.parameters.loss_function_parameters.get(rp)

        self.model_leverage = self.parameters.get('leverage', 1.0)
        #
        model_config = dict(num_inputs=self.num_inputs,
                            num_outputs=self.num_outputs,
                            num_hidden=self.num_hidden,
                            initial_wealth=self.parameters.initial_wealth,
                            learning_rate=self.parameters.initial_learning_rate,
                            leverage=self.model_leverage,
                            device='cpu')

        self.control_model = model(model_config, returns)
        self.epoch_data = []
        #
        self.validate_executor_state()

    def validate_executor_state(self):
        assert self.control_model is not None
        assert self.epoch_data is not None
        assert self.num_inputs == 2
        assert self.num_features == 0
        assert self.num_benchmarks == 0

    def generate_simulated_datasets(self, num_train_paths, num_validation_paths, num_test_paths, horizon):
        """
        Generate brownian motions and feed them to the model
        """

        N = horizon + 1
        train_price_paths = generate_simulated_price_paths(self.spots,
                                                           self.uncorrelated_gbms,
                                                           self.maturity,
                                                           N,
                                                           num_train_paths,
                                                           self.correlation_matrix)

        validation_price_paths = generate_simulated_price_paths(self.spots,
                                                                self.uncorrelated_gbms,
                                                                self.maturity,
                                                                N,
                                                                num_validation_paths,
                                                                self.correlation_matrix)

        test_price_paths = generate_simulated_price_paths(self.spots,
                                                          self.uncorrelated_gbms,
                                                          self.maturity,
                                                          N,
                                                          num_test_paths,
                                                          self.correlation_matrix)

        train_paths = []
        validation_paths = []
        test_paths = []

        for j in range(num_train_paths):
            current_path = train_price_paths[:, j, :]
            df = pd.DataFrame(current_path.transpose())
            df.columns = ['asset_' + str(i + 1) for i in range(self.num_assets)]
            returns = (df.diff() / df).dropna(inplace=False)
            returns.index = np.arange(0, self.maturity, self.maturity / horizon)
            returns.columns = ['asset_' + str(i + 1) for i in range(self.num_assets)]
            assert isinstance(returns, pd.DataFrame)
            assert len(returns) == horizon
            train_paths.append(returns.to_numpy())
            # print('train paths', train_paths)

        for j in range(num_validation_paths):
            current_path = validation_price_paths[:, j, :]
            df = pd.DataFrame(current_path.transpose())
            df.columns = ['asset_' + str(i + 1) for i in range(self.num_assets)]
            returns = (df.diff() / df).dropna(inplace=False)
            returns.index = np.arange(0, self.maturity, self.maturity / horizon)
            returns.columns = ['asset_' + str(i + 1) for i in range(self.num_assets)]
            assert isinstance(returns, pd.DataFrame)
            validation_paths.append(returns.to_numpy())

        for j in range(num_test_paths):
            current_path = test_price_paths[:, j, :]
            df = pd.DataFrame(current_path.transpose())
            df.columns = ['asset_' + str(i + 1) for i in range(self.num_assets)]
            returns = (df.diff() / df).dropna(inplace=False)
            returns.index = np.arange(0, self.maturity, self.maturity / horizon)
            returns.columns = ['asset_' + str(i + 1) for i in range(self.num_assets)]
            assert isinstance(returns, pd.DataFrame)
            test_paths.append(returns.to_numpy())
            # print('test paths', test_paths)


        # print(train_paths)
        return train_paths, validation_paths, test_paths

    def execute(self):

        # parameters
        num_epochs = self.parameters.max_num_epochs
        num_train_paths = self.parameters.num_train_paths
        num_validation_paths = self.parameters.num_validation_paths
        num_test_paths = self.parameters.num_test_paths
        horizon = self.parameters.horizon  # investment horizon

        if self.requires_target_wealth:
            target_wealth = self.parameters.target_wealth
        else:
            target_wealth = None
        initial_wealth = self.parameters.initial_wealth
        assert isinstance(self.control_model, BaseControlModel)

        ################################################################################################################
        # generate training, validation and test paths
        train_paths, validation_paths, test_paths = self.generate_simulated_datasets(num_train_paths,
                                                                                     num_validation_paths,
                                                                                     num_test_paths,
                                                                                     horizon)

        ################################################################################################################
        # set up target wealth vector and some variables needed for the training loop
        ################################################################################################################
        if self.requires_target_wealth:
            target_wealths = torch.from_numpy(target_wealth * np.ones(num_train_paths)).float().to(
                self.control_model.device)
        else:
            target_wealths = None
        if num_validation_paths > 0:
            if self.requires_target_wealth:
                target_wealths_validation = torch.from_numpy(target_wealth * np.ones(num_validation_paths)).float().to(
                    self.control_model.device)
            else:
                target_wealths_validation = None
        else:
            target_wealths_validation = None

        self.control_model.model.to(self.control_model.device)
        logger.info('[ModelExecutor] selected device: {}'.format(self.control_model.device))
        logger.info('[ModelExecutor] selected loss function: {}'.format(self.loss_function.__name__))
        logger.info('[ModelExecutor] training started')
        self.control_model.model.train()
        stop_training = False
        prev_training_loss = None
        prev_validation_loss = None
        consecutive_validation_loss_increases = []

        ################################################################################################################
        # training loop
        ################################################################################################################
        for k in range(num_epochs):

            if stop_training:
                # If this flag was set to True during a training epoch, the training stops
                # This can occur if the training loss or validation loss start increasing and
                # the optimization cannot recover (i.e. the increase in loss was not temporary jitter)
                break

            logger.info('[ModelExecutor] epoch {} started'.format(k + 1))
            path_wealth_list = []
            epoch_start = dt.datetime.now()
            self.control_model.model.train()

            for l, path in enumerate(train_paths):
                final_path_wealths = process_train_path(self.control_model, path, self.num_features,
                                                        self.num_benchmarks, verbose=False)
                path_wealth_list.append(final_path_wealths)

            path_wealth_tensor = torch.stack(path_wealth_list).float()

            if target_wealths is not None:
                if self.loss_function_parameters:
                    loss = self.loss_function(path_wealth_tensor, target_wealths, **self.loss_function_parameters)
                else:
                    loss = self.loss_function(path_wealth_tensor, target_wealths)
            else:
                if self.loss_function_parameters:
                    loss = self.loss_function(path_wealth_tensor, None, **self.loss_function_parameters)
                else:
                    loss = self.loss_function(path_wealth_tensor, None)
            detached_wealth_array = path_wealth_tensor.detach().cpu().numpy()

            training_wealths_stats = compute_wealth_distribution_statistics(detached_wealth_array,
                                                                            initial_wealth=initial_wealth,
                                                                            target_wealth=target_wealth)
            average_epoch_wealth = training_wealths_stats.mean
            median_epoch_wealth = training_wealths_stats.median
            min_epoch_wealth = training_wealths_stats.min
            max_epoch_wealth = training_wealths_stats.max
            fraction_above_target = training_wealths_stats.above_target_wealth
            fraction_above_initial = training_wealths_stats.above_initial_wealth

            # compute gradients and zero the gradients
            loss.backward()
            self.control_model.optimizer.step()
            self.control_model.optimizer.zero_grad()

            epoch_end = dt.datetime.now()
            epoch_time = (epoch_end - epoch_start).total_seconds()

            # check performance on validation set
            validation_wealths = []
            validation_controls = []
            if num_validation_paths > 0:
                for validation_path in tqdm(validation_paths, desc='[ModelExecutor] evaluating validation performance'):
                    w, w_path, v_control, _ = self.control_model.compute_path_wealth_and_controls(validation_path,
                                                                                                  self.num_features,
                                                                                                  self.num_benchmarks)

                    validation_wealths.append(w)
                    validation_controls.append(v_control)

                validation_paths_stats = compute_wealth_distribution_statistics(validation_wealths,
                                                                                initial_wealth=initial_wealth,
                                                                                target_wealth=target_wealth)
            else:
                validation_paths_stats = None
            validation_path_wealth_tensor = torch.tensor(validation_wealths).float()
            if num_validation_paths > 0:
                if self.loss_function_parameters:
                    validation_loss = self.loss_function(validation_path_wealth_tensor, target_wealths_validation,
                                                         **self.loss_function_parameters)
                else:
                    validation_loss = self.loss_function(validation_path_wealth_tensor, target_wealths_validation)
            else:
                validation_loss = None

            # print epoch stats
            if validation_loss:
                logger.info(
                    '[ModelExecutor] epoch: {}, Train Loss: {:.8f}, Validation Loss: {:.8f}'.format(
                        k + 1,
                        loss,
                        validation_loss))
            else:
                logger.info(
                    '[ModelExecutor] epoch: {}, Train Loss: {:.8f}'.format(k + 1, loss))

            logger.info('--->  training stats -> avg: {:.4f}, median: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
                average_epoch_wealth,
                median_epoch_wealth,
                min_epoch_wealth,
                max_epoch_wealth))

            logger.info(
                '--->  Prob(W_T > W_*: {}, Prob (W_T > 1.0): {} || compute time: {:.2f}'.format(fraction_above_target,
                                                                                                fraction_above_initial,
                                                                                                epoch_time))

            if validation_paths_stats:
                logger.info('---> validation stats -> avg: {:.4f}, median: {:.4f}, min: {:.4f}, max: {:.4f}'.format(
                    validation_paths_stats.mean,
                    validation_paths_stats.median,
                    validation_paths_stats.min,
                    validation_paths_stats.max))

            epoch_info = {
                'epoch': k + 1,
                'training_loss': loss.detach().numpy(),
                'validation_loss': validation_loss,
                'fraction_above_target': fraction_above_target,
                'fraction_above_initial': fraction_above_initial
            }

            self.epoch_data.append(epoch_info)

            if prev_training_loss:
                logger.info('[ModelExecutor] prev_training_loss: {}'.format(prev_training_loss))
                if loss.detach().numpy() >= prev_training_loss:
                    self.control_model.update_learning_rate()
            prev_training_loss = loss.detach().numpy()

            # early stopping is activated after min_num_epochs and stops training if
            # the validation loss starts to increase
            if k > self.parameters.min_num_epochs:
                if prev_validation_loss:
                    if validation_loss > prev_validation_loss:
                        consecutive_validation_loss_increases.append(k)

                    if len(consecutive_validation_loss_increases) > 3:
                        stop_training = True

            prev_validation_loss = validation_loss

        ################################################################################################################
        # end training loop
        ################################################################################################################

        logger.info('[ModelExecutor] training completed\n')
        logger.info('[ModelExecutor] processing results')

        test_paths_final_wealths = []
        test_paths_wealth_trajectories = []  # evolution of the wealth for each path
        test_paths_controls = []
        test_paths_benchmark_wealths = []

        ################################################################################################################
        # generate results
        ################################################################################################################

        # recompute path wealths along training paths ###########################################################
        train_paths_controls = []
        train_paths_wealth_trajectories = []
        train_paths_final_wealths = []

        validation_paths_controls = []
        validation_paths_wealth_trajectories = []
        validation_paths_final_wealths = []

        logger.info('[ModelExecutor] evaluating model on train paths')
        # use the final trained model to compute the wealths and controls along the training and validation paths
        for train_path in train_paths:
            w, w_path, w_controls, _ = self.control_model.compute_path_wealth_and_controls(train_path,
                                                                                           self.num_features,
                                                                                           self.num_benchmarks)
            train_paths_final_wealths.append(w)
            train_paths_wealth_trajectories.append(w_path)
            train_paths_controls.append(w_controls)

        logger.info('[ModelExecutor] evaluating model on validation paths')
        for validation_path in validation_paths:
            w, w_path, w_controls, _ = self.control_model.compute_path_wealth_and_controls(validation_path,
                                                                                           self.num_features,
                                                                                           self.num_benchmarks)
            validation_paths_final_wealths.append(w)
            validation_paths_wealth_trajectories.append(w_path)
            validation_paths_controls.append(w_controls)

        logger.info('[ModelExecutor] evaluating model on test paths')
        for test_path in test_paths:
            w, w_path, w_controls, benchmark_wealth = self.control_model.compute_path_wealth_and_controls(test_path,
                                                                                                          self.num_features,
                                                                                                          self.num_benchmarks)
            test_paths_final_wealths.append(w)
            test_paths_benchmark_wealths.append(benchmark_wealth)
            # print(w, benchmark_wealth)
            # test_paths_strategy_excess_wealths_over_benchmark.append(w - benchmark_wealth)
            test_paths_wealth_trajectories.append(w_path)
            test_paths_controls.append(w_controls)

        # logger.info('[ModelExecutor] evaluating model on historical test path')
        # # actual test data - historical backtest
        # historical_test_path = self.test_data.to_numpy()
        # w_backtest, w_path_backtest, w_controls_backtest, _ = self.control_model.compute_path_wealth_and_controls(
        #     historical_test_path,
        #     self.num_features,
        #     self.num_benchmarks)

        ################################################################################################################

        logger.info('[ModelExecutor] computing wealth distribution statistics')
        train_wealths_stats = compute_wealth_distribution_statistics(np.array(train_paths_final_wealths),
                                                                     initial_wealth=initial_wealth,
                                                                     target_wealth=target_wealth)
        validation_wealths_stats = compute_wealth_distribution_statistics(np.array(validation_paths_final_wealths),
                                                                          initial_wealth=initial_wealth,
                                                                          target_wealth=target_wealth)
        test_wealths_stats = compute_wealth_distribution_statistics(np.array(test_paths_final_wealths),
                                                                    initial_wealth=initial_wealth,
                                                                    target_wealth=target_wealth)

        epoch_training_losses = [edata['training_loss'] for edata in self.epoch_data]
        epoch_validation_losses = [edata['validation_loss'] for edata in self.epoch_data]
        epochs = [i for i, _ in enumerate(self.epoch_data)]

        logger.info('[ModelExecutor] constructing result object')

        result_set = {
            'config': {
                'assets': self.assets
            },
            'training': {
                'return_paths': train_paths,
                'controls': train_paths_controls,
                'final_wealth': train_paths_final_wealths,
                'wealth_trajectories': train_paths_wealth_trajectories,
                'stats': train_wealths_stats,
                'loss': epoch_training_losses,
                'epochs': epochs
            },

            'validation': {
                'return_paths': validation_paths,
                'controls': validation_paths_controls,
                'final_wealth': validation_paths_final_wealths,
                'wealth_trajectories': validation_paths_wealth_trajectories,
                'stats': validation_wealths_stats,
                'loss': epoch_validation_losses,
                'epochs': epochs
            },

            'test': {
                'return_paths': test_paths,
                'controls': test_paths_controls,
                'final_wealth': test_paths_final_wealths,
                'wealth_trajectories': test_paths_wealth_trajectories,
                'stats': test_wealths_stats,
                # 'backtest': {
                #     'final_wealth': w_backtest,
                #     'wealth_trajectories': w_path_backtest,
                #     'controls': w_controls_backtest
                # }
            }

        }

        result_object = munch.Munch.fromDict(result_set)
        logger.info('[ModelExecutor] done ---------------------')

        return result_object
