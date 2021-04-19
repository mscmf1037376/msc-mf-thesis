import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf as ACF
from statsmodels.tsa.stattools import acovf
from statsmodels.graphics.tsaplots import plot_acf


def R_hat(data: np.ndarray, k: int):
    """
    Estimate of the autocovariance function
    data will have several columns (asset returns)
    """

    assert isinstance(k, int)
    assert len(data.shape) == 1
    N = data.shape[0]
    assert abs(k) < N
    mean = data.mean(axis=0)
    sum_variances = 0

    for i in range(1, N - abs(k) + 1):
        var = (data[i - 1] - mean) * (data[i - 1 + abs(k)])
        sum_variances += var

    return sum_variances / N


def rho_hat(data: np.ndarray, k: int) -> float:
    """

    :param data:
    :param k:
    :return:
    """
    R_hat_0 = R_hat(data, 0)
    R_hat_k = R_hat(data, k)

    if R_hat_k <= 1e-3:
        return 0

    return R_hat_k / R_hat_0


def trapezoidal_window(t: float) -> float:
    """
    In Politis & White (2004) paper
    Automatic Block-Length Selection for the Dependent Bootstrap
    :param t:
    :return:
    """

    abs_t = abs(t)

    if (abs_t >= 0) and (abs_t <= 0.5):
        return 1.0
    elif (abs_t > 0.5) and (abs_t <= 1):
        return 2.0 * (1 - abs_t)
    else:
        return 0.0


def g_hat(data: np.array, M: int, omega: float) -> float:
    """
    Estimate of the power spectral density function
    :param data:
    :param M:
    :param k:
    :return:
    """
    assert isinstance(M, int)
    assert len(data.shape) == 1
    assert M < data.shape[0]

    result = 0

    for k in range(-M, M + 1):
        val = trapezoidal_window((float(k) / M)) * R_hat(data, k) * np.cos(float(omega) * k)
        result += val

    return result


def G_hat(data: np.array, M: int) -> float:
    """
    Compute an estimate of the quantity G in Politis & White 2004
    G =         Sum  [ abs(k) * R(k) ]
        k from -inf to + inf

    R(k) is the autocovariance function, approximated by R_hat above.
    :param data:
    :param M:
    :return:
    """
    assert isinstance(M, int)
    assert len(data.shape) == 1
    assert M < data.shape[0]
    result = 0

    for k in range(-M, M + 1):
        g_k = abs(k) * trapezoidal_window(k / M) * R_hat(data, k)
        result += g_k

    return result


def get_optimal_block_size_stationary_bootstrap(data: np.array) -> int:
    """
    Compute the optimal block size for stationary bootstrap
    :param data: 1D numpy array
    :return: optimal block size
    """

    assert len(data.shape) == 1
    N = data.shape[0]
    M = int(N / 10)
    D_SB = 2 * g_hat(data, M, 0)

    G_hat_val = G_hat(data, M)

    # compute this intermediate term
    intermediate_term = 2 * G_hat_val * G_hat_val / D_SB
    # due to floating point precision, this term can become negative if it's very close to zero
    intermediate_term = max(0.0, intermediate_term)
    b_opt = (intermediate_term ** (1 / 3)) * N ** (1 / 3)
    return max(1, int(round(b_opt)))


def get_optimal_block_size_circular_bootstrap(data: np.array) -> int:
    """
    Compute the optimal block size for circular bootstrap
    :param data: 1D numpy array
    :return: optimal block size
    """
    assert len(data.shape) == 1
    N = data.shape[0]
    M = int(N / 10)
    D_CB = (4.0 / 3.0) * g_hat(data, M, 0)
    G_hat_val = G_hat(data, M)

    # compute this intermediate term
    intermediate_term = 2 * G_hat_val * G_hat_val / D_CB
    # due to floating point precision, this term can become negative if it's very close to zero
    intermediate_term = max(0.0, intermediate_term)
    b_opt = (intermediate_term ** (1 / 3)) * N ** (1 / 3)
    return max(1, int(round(b_opt)))
