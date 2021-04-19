import functools
from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch

from wealth_optimizer.common_logger import logger


def validate_control_model(func):
    """
    Decorator for validating that the control model is initialized correctly
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        self_ = args[0]
        assert self_ is not None
        assert hasattr(self_, 'model')
        assert hasattr(self_, 'device')
        assert hasattr(self_, 'optimizer')
        assert hasattr(self_, 'initial_wealth')

    return wrapper


class ModelSelector(object):

    def __init__(self):
        self.model_mapping = {
            'long_only': LongOnlyLeveragedOneLayerNetwork,
            'long_short': LongShortLeveragedOneLayerNetwork
        }

    def get_model(self, model_name: str) -> Callable:
        if model_name not in self.model_mapping.keys():
            raise ValueError('Model not found: {}'.format(model_name))
        return self.model_mapping[model_name]


class BaseControlModel(ABC):
    """
    Base class for different model structures
    """

    def __init__(self, config: object, data: pd.DataFrame):
        if not config:
            raise ValueError('must pass a config to the model')
        if data is None:
            raise ValueError('must pass a data frame to the model')
        self.model = None
        self.num_features, self.has_benchmark = self.__extract_column_info(data)
        self.num_benchmarks = 1 if self.has_benchmark else 0
        logger.info('initializing control model with num_features: {}, num_benchmarks: {}'.format(self.num_features,
                                                                                                  self.num_benchmarks))

    @abstractmethod
    def update_learning_rate(self, *args, **kwargs):
        """
        Abstract method for updating the learning rate of the optimizer
        """
        raise NotImplementedError

    @abstractmethod
    def compute_path_states(self, *args, **kwargs):
        """
        Abstract method for computing the state (current time, current wealth and other features) along sample paths
        """
        raise NotImplementedError

    @abstractmethod
    def compute_path_wealth_and_controls(self, *args, **kwargs):
        """
        Abstract method for computing the wealths and controls along each sample path once the model is trained
        """
        raise NotImplementedError

    @staticmethod
    def __extract_column_info(data: pd.DataFrame) -> Tuple[int, bool]:
        """
        Checks if the dataset has a benchmark column and features columns
        The generic model doesn't require any input features, but they can be added as columns
        If a benchmark is present, the executor will generate sample paths from the benchmark in the same way it
        generates them for the assets and features. Then it will compute the performance of the benchmark asset
        against the learned control strategy.
        :param data:
        :return:
        """
        columns = data.columns
        has_benchmark = False
        num_features = 0
        for col in columns:
            if 'feature_' in col:
                num_features += 1
            if 'benchmark_' in col:
                has_benchmark = True

        return num_features, has_benchmark


class LeverageLayer(torch.nn.Module):
    def __init__(self, leverage: float):
        super().__init__()
        self.leverage = leverage
        logger.info('[LeverageLayer] leverage: {}'.format(self.leverage))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.leverage * inputs


class ScaledTanh(torch.nn.Module):
    def __init__(self, leverage: float):
        super().__init__()
        self.leverage = leverage
        logger.info('[ScaledTanh] leverage: {}'.format(self.leverage))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        num_assets = len(inputs)
        pos_limit = 1.0 / num_assets
        return self.leverage * pos_limit * torch.tanh(inputs)


class LongOnlyLeveragedOneLayerNetwork(BaseControlModel):

    @validate_control_model
    def __init__(self, config: dict, data: pd.DataFrame):
        super().__init__(config, data)
        self.num_inputs = config['num_inputs']
        self.num_outputs = config['num_outputs']
        self.num_hidden = config['num_hidden']
        self.learning_rate = config['learning_rate']
        self.device = config['device']
        self.leverage = config.get('leverage', 1.0)
        self.initial_wealth = config['initial_wealth']
        self.num_assets = None

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.num_inputs, self.num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_hidden, self.num_outputs),
            torch.nn.Softmax(dim=0),
            LeverageLayer(self.leverage)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_learning_rate(self):
        logger.info('[update_learning_rate] current learning rate: {}'.format(self.learning_rate))
        self.learning_rate /= 5
        logger.info('[update_learning_rate] updated learning rate: {}'.format(self.learning_rate))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def compute_path_states(self, path: np.ndarray, num_features: int, num_benchmarks: int,
                            verbose: bool = False) -> np.array:
        """

        :param path:
        :param num_features:
        :param num_benchmarks:
        :param verbose:
        :return:
        """
        self.model.eval()  # switch model to eval mode
        column_offset = num_features + num_benchmarks
        path_wealth = self.initial_wealth

        benchmark_wealth = self.initial_wealth
        path_length = len(path)
        use_benchmark = num_benchmarks > 0

        if use_benchmark:

            first_row_features = list(path[0, 0:column_offset])
        else:
            first_row_features = list(path[0, :num_features])

        if use_benchmark:
            state0 = np.array([0.0, path_wealth, *first_row_features], dtype=np.float32)
        else:
            state0 = np.array([0.0, path_wealth, *first_row_features], dtype=np.float32)
        path_states = [state0]
        for n in range(1, path_length):
            to_go = (path_length - n) / path_length
            test_data = torch.from_numpy(path_states[n - 1]).to(self.device)
            allocations = self.model(test_data).detach().cpu().numpy()

            portfolio_return = np.dot(path[n][column_offset:], allocations)
            path_wealth *= 1 + portfolio_return

            if use_benchmark:
                benchmark_return = float(path[n][num_features])
                benchmark_wealth *= 1 + benchmark_return

            if verbose:
                logger.info(
                    '[compute_path_states][{}/{}]: asset returns: {}'.format(n, path_length, path[n][column_offset:]))
                logger.info('[compute_path_states][{}/{}]: asset weights: {}'.format(n, path_length, allocations))
                logger.info(
                    "[compute_path_states][{}/{}]: portfolio_return: {}".format(n, path_length, portfolio_return))
                logger.info("[compute_path_states][{}/{}]: path_wealth: {}".format(n, path_length, path_wealth))

                if use_benchmark:
                    logger.info(
                        "[compute_path_states][{}/{}]: benchmark_wealth: {}".format(n, path_length, benchmark_wealth))
            if path_wealth < 0:
                raise ValueError('[compute_path_states] path wealth cannot be negative')

            if use_benchmark:
                path_states.append(np.array([to_go, path_wealth, *list(path[n][:column_offset])], dtype=np.float32))
            else:
                path_states.append(np.array([to_go, path_wealth, *list(path[n][:num_features])], dtype=np.float32))
        self.model.train()  # switch back to training mode
        return np.array(path_states)

    def compute_path_wealth_and_controls(self, path: np.ndarray, num_features: int, num_benchmarks: int):
        """

        :param path:
        :param num_features:
        :param num_benchmarks:
        :return:
        """
        self.model.eval()
        final_path_wealth = self.initial_wealth
        benchmark_wealth = self.initial_wealth
        use_benchmark = num_benchmarks > 0
        column_offset = num_features + num_benchmarks
        path_length = len(path)

        if use_benchmark:
            first_row_features = list(path[0, 0:column_offset])
            state0 = np.array([0.0, final_path_wealth, *first_row_features], dtype=np.float32)
        else:
            first_row_features = list(path[0, :num_features])
            state0 = np.array([0.0, final_path_wealth, *first_row_features], dtype=np.float32)


        path_states = [state0]
        path_wealths = [self.initial_wealth]
        controls = []
        for n in range(1, path_length):
            to_go = (path_length - n) / path_length
            test_data = torch.from_numpy(path_states[n - 1]).to(self.device)
            # print(test_data)
            control = self.model(test_data).detach().cpu().numpy()
            portfolio_return = np.dot(path[n][column_offset:], control)
            final_path_wealth *= 1 + portfolio_return
            if final_path_wealth < 0:
                raise ValueError('[compute_test_path_wealth] path wealth cannot be negative')

            if use_benchmark:
                benchmark_return = float(path[n][num_features])
                benchmark_wealth *= 1 + benchmark_return

            if use_benchmark:
                path_states.append(
                    np.array([to_go, final_path_wealth, *list(path[n][:column_offset])], dtype=np.float32))
            else:
                path_states.append(
                    np.array([to_go, final_path_wealth, *list(path[n][:num_features])], dtype=np.float32))

            path_wealths.append(final_path_wealth)
            controls.append(control)

        return final_path_wealth, path_wealths, controls, benchmark_wealth


class LongShortLeveragedOneLayerNetwork(BaseControlModel):

    @validate_control_model
    def __init__(self, config: dict, data: pd.DataFrame):
        super().__init__(config, data)
        self.num_inputs = config['num_inputs']
        self.num_outputs = config['num_outputs']
        self.num_hidden = config['num_hidden']
        self.learning_rate = config['learning_rate']
        self.device = config['device']
        self.initial_wealth = config['initial_wealth']
        self.leverage = config.get('leverage', 1.0)
        self.num_assets = None

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.num_inputs, self.num_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.num_hidden, self.num_outputs),
            ScaledTanh(self.leverage)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_learning_rate(self):
        logger.info('[LongShortLeveragedOneLayerNetwork][update_learning_rate] current learning rate: {}'.format(
            self.learning_rate))
        self.learning_rate /= 5
        logger.info('[LongShortLeveragedOneLayerNetwork][update_learning_rate] updated learning rate: {}'.format(
            self.learning_rate))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def compute_path_states(self, path: np.ndarray, num_features: int, num_benchmarks: int,
                            verbose: bool = False) -> np.array:
        """

        :param path:
        :param num_features:
        :param num_benchmarks:
        :param verbose:
        :return:
        """
        self.model.eval()  # switch model to eval mode
        column_offset = num_features + num_benchmarks
        path_wealth = self.initial_wealth

        benchmark_wealth = self.initial_wealth
        path_length = len(path)
        use_benchmark = num_benchmarks > 0

        if use_benchmark:

            first_row_features = list(path[0, 0:column_offset])
        else:
            first_row_features = list(path[0, :num_features])

        if use_benchmark:
            state0 = np.array([0.0, path_wealth, *first_row_features], dtype=np.float32)
        else:
            state0 = np.array([0.0, path_wealth, *first_row_features], dtype=np.float32)
        path_states = [state0]
        for n in range(1, path_length):
            to_go = (path_length - n) / path_length
            test_data = torch.from_numpy(path_states[n - 1]).to(self.device)
            allocations = self.model(test_data).detach().cpu().numpy()

            portfolio_return = np.dot(path[n][column_offset:], allocations)
            path_wealth *= 1 + portfolio_return

            if use_benchmark:
                benchmark_return = float(path[n][num_features])
                benchmark_wealth *= 1 + benchmark_return

            if verbose:
                logger.info(
                    '[compute_path_states][{}/{}]: asset returns: {}'.format(n, path_length, path[n][column_offset:]))
                logger.info('[compute_path_states][{}/{}]: asset weights: {}'.format(n, path_length, allocations))
                logger.info(
                    "[compute_path_states][{}/{}]: portfolio_return: {}".format(n, path_length, portfolio_return))
                logger.info("[compute_path_states][{}/{}]: path_wealth: {}".format(n, path_length, path_wealth))

                if use_benchmark:
                    logger.info(
                        "[compute_path_states][{}/{}]: benchmark_wealth: {}".format(n, path_length, benchmark_wealth))
            if path_wealth < 0:
                raise ValueError('[compute_path_states] path wealth cannot be negative')

            if use_benchmark:
                path_states.append(np.array([to_go, path_wealth, *list(path[n][:column_offset])], dtype=np.float32))
            else:
                path_states.append(np.array([to_go, path_wealth, *list(path[n][:num_features])], dtype=np.float32))
        self.model.train()  # switch back to training mode
        return np.array(path_states)

    def compute_path_wealth_and_controls(self, path: np.ndarray, num_features: int, num_benchmarks: int):
        """

        :param path:
        :param num_features:
        :param num_benchmarks:
        :return:
        """
        self.model.eval()
        final_path_wealth = self.initial_wealth
        benchmark_wealth = self.initial_wealth
        use_benchmark = num_benchmarks > 0
        column_offset = num_features + num_benchmarks
        path_length = len(path)

        if use_benchmark:
            first_row_features = list(path[0, 0:column_offset])
            state0 = np.array([0.0, final_path_wealth, *first_row_features], dtype=np.float32)

        else:
            first_row_features = list(path[0, :num_features])
            state0 = np.array([0.0, final_path_wealth, *first_row_features], dtype=np.float32)

        path_states = [state0]
        path_wealths = [self.initial_wealth]
        controls = []
        for n in range(1, path_length):
            to_go = (path_length - n) / path_length
            test_data = torch.from_numpy(path_states[n - 1]).to(self.device)
            control = self.model(test_data).detach().cpu().numpy()
            portfolio_return = np.dot(path[n][column_offset:], control)
            final_path_wealth *= 1 + portfolio_return
            if final_path_wealth < 0:
                raise ValueError('[compute_test_path_wealth] path wealth cannot be negative')

            if use_benchmark:
                benchmark_return = float(path[n][num_features])
                benchmark_wealth *= 1 + benchmark_return

            if use_benchmark:
                path_states.append(
                    np.array([to_go, final_path_wealth, *list(path[n][:column_offset])], dtype=np.float32))
            else:
                path_states.append(
                    np.array([to_go, final_path_wealth, *list(path[n][:num_features])], dtype=np.float32))

            path_wealths.append(final_path_wealth)
            controls.append(control)

        return final_path_wealth, path_wealths, controls, benchmark_wealth
