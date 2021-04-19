from typing import Callable, List, Union
import numpy as np

# https://www.goddardconsulting.ca/matlab-monte-carlo-assetpaths-corr.html
# https://mikejuniperhill.blogspot.com/2019/04/python-path-generator-for-correlated.html


def geometric_brownian_motion_generator(drift: float, volatility: float) -> Callable:
    """
    Represents standard Brownian Motion with given drift, volatility, current value and time step dt
    :param drift: drift
    :param volatility: volatility
    :return: the value of the process at the next time step (t+dt)
    """

    def parametrized_gbm(s, dt, z):
        """
        :param s: current price
        :param dt: time increment
        :param z: sample from a standard normal distribution
        """

        adj_drift = drift - 0.5 * volatility * volatility
        exponent = adj_drift * dt + volatility * np.sqrt(dt) * z
        return s * np.exp(exponent)

    return parametrized_gbm


def generate_simulated_price_paths(spot: np.ndarray, processes: Union[Callable, List[Callable]], maturity: float,
                                   n_steps,
                                   n_paths, correlation: np.ndarray):
    """
    Generates simulated price paths based on correlated or uncorrelated Brownian motions
    :param spot: initial values of the price processes
    :param processes: list of generator functions for each process
    :param maturity: maturity in years
    :param n_steps: number of samples to generate
    :param n_paths: number of sample paths
    :param correlation: correlation matrix
    :return: returns numpy array with the following dimensions: nProcesses, nPaths, nSteps
    """
    # validate input parameters
    assert isinstance(spot, np.ndarray)
    assert correlation is not None
    assert isinstance(correlation, np.ndarray)

    dt = maturity / n_steps

    # case: given correlation matrix, create paths for multiple correlated processes
    if not isinstance(correlation, np.ndarray):
        raise ValueError('no correlation provided')
    n_processes = len(processes)
    result = np.zeros(shape=(n_processes, n_paths, n_steps))

    # loop through number of paths
    for i in range(n_paths):
        # create one set of correlated random variates for n processes
        cholesky_matrix = np.linalg.cholesky(correlation)
        e = np.random.normal(size=(n_processes, n_steps))
        paths = np.dot(cholesky_matrix, e)
        # loop through number of steps
        for j in range(n_steps):
            # loop through number of processes
            for k in range(n_processes):
                # first path value is always current spot price
                if j == 0:
                    result[k, i, j] = paths[k, j] = spot[k]
                else:
                    # use SDE lambdas (inputs: previous spot, dt, current random variate)
                    result[k, i, j] = paths[k, j] = processes[k](paths[k, j - 1], dt, paths[k, j])

    return result
