"""Loss functions used in model training"""
from typing import Callable, List, Tuple, Optional
import torch


class LossFunctionSelector(object):
    """
    A class that stores and selects loss function based on a passed string
    """

    def __init__(self):
        self.loss_mapping = {
            'forsyth_li_regularized_loss': {'function': forsyth_li_regularized_loss, 'requires_target': True,
                                            'parameters': None},
            'quadratic_loss_linear_gain_penalty': {'function': quadratic_loss_linear_gain_penalty,
                                                   'requires_target': True, 'parameters': None},
            'biquadratic_regularized_loss': {'function': biquadratic_regularized_loss, 'requires_target': True,
                                             'parameters': None},
            'regularized_max_loss': {'function': regularized_max_loss, 'requires_target': True,
                                     'parameters': ['lambda_reg']},
            'max_loss': {'function': max_loss, 'requires_target': True},
            'sharpe_ratio_loss': {'function': sharpe_ratio_loss, 'requires_target': True},
            'mean_log_utility': {'function': mean_log_utility, 'requires_target': False, 'parameters': None},
            'quantile_log_utility': {'function': quantile_log_utility, 'requires_target': False,
                                     'parameters': ['quantile']},
            'power_utility': {'function': power_utility, 'requires_target': False, 'parameters': ['p']},

        }

    def get_loss_function(self, loss_func: str) -> Tuple[Callable, bool]:
        if loss_func not in self.loss_mapping.keys():
            raise ValueError('Loss function not found: {}'.format(loss_func))
        return self.loss_mapping[loss_func]['function'], self.loss_mapping[loss_func]['requires_target']

    def get_required_parameters(self, loss_func: str) -> Optional[List]:
        if loss_func not in self.loss_mapping.keys():
            raise ValueError('Loss function not found: {}'.format(loss_func))
        return self.loss_mapping[loss_func]['parameters']


def mean_log_utility(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Based on the log utility function
    U(x) = log(x)

    Loss = - E[U(x)] = - E[log(x)] over a list of final wealth values
    :param output: tensor of final wealths
    :param target: NOT USED
    :return: value of loss function
    """
    loss = - torch.mean(torch.log(output))
    return loss


def quantile_log_utility(output: torch.Tensor, target: torch.Tensor, quantile: float = 0.2) -> torch.Tensor:
    """
    Same as the log utility but instead of mean, we take a quantile (i.e worst q %) rather than mean
    :param output: tensor of final wealths
    :param target: NOT USED
    :param quantile: quantile of distribution
    :return: value of loss function
    """
    loss = - torch.quantile(torch.log(output), q=quantile)
    return loss


def power_utility(output: torch.Tensor, target: torch.Tensor, p: float) -> torch.Tensor:
    """
    Power utility loss function#
    U(x) = x ^ p / p, where 0 < p < 1
    This loss function returns -U(x)
    :param output: tensor of final wealths
    :param target: NOT USED
    :param p: power 0 < p < 1
    :return: value of loss function
    """
    assert p > 0.0
    assert p < 1.0

    loss = - torch.mean(torch.pow(output, p))
    return loss


def sharpe_ratio_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute a Sharpe ratio between excess return (above target) and volatility of excess return
    :param output: tensor of final wealths
    :param target: tensor of target wealths
    :return: loss
    """

    excess_returns = output - target
    mean_excess_return = torch.mean(excess_returns)
    std_excess_return = torch.std(excess_returns)
    print('mean_excess_return', mean_excess_return)
    print('std_excess_return', std_excess_return)
    sharpe_ratio = mean_excess_return / std_excess_return
    return -sharpe_ratio


def forsyth_li_regularized_loss(output: torch.Tensor, target: torch.Tensor, lambda_reg: float = 1e-6) -> torch.Tensor:
    """
    As per Forsyth & Li paper:
    A Data Driven Neural Network Approach to Optimal Asset Allocation for
    Target Based Defined Contribution Pension Plans, June 2018
    :param output: output wealths
    :param target: target wealths
    :param lambda_reg: regularization constant
    :return: loss
    """
    mean_sq_undershoot = torch.mean(torch.clamp(target - output, min=0.0) ** 2)
    regularizing_term = lambda_reg * torch.mean(output)
    loss = mean_sq_undershoot + regularizing_term

    return loss


def quadratic_loss_linear_gain_penalty(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    As per Forsyth & Li paper:
    A Data Driven Neural Network Approach to Optimal Asset Allocation for
    Target Based Defined Contribution Pension Plans, June 2018
    :param output: output wealths
    :param target: target wealths
    :return: loss
    """
    penalty = torch.mean(torch.clamp(target - output, min=0.0) ** 2)
    reward = torch.mean(torch.clamp(output - target, min=0.0))
    loss = penalty + reward

    return loss


def biquadratic_regularized_loss(output: torch.Tensor, target: torch.Tensor, lambda_reg: float = 1e-6) -> torch.Tensor:
    """
    As per Forsyth & Li paper:
    A Data Driven Neural Network Approach to Optimal Asset Allocation for
    Target Based Defined Contribution Pension Plans, June 2018
    :param output: output wealths
    :param target: target wealths
    :param lambda_reg: regularization constant
    :return: loss
    """
    mean_sq_undershoot = torch.mean(torch.clamp(target - output, min=0.0) ** 4)
    regularizing_term = lambda_reg * torch.mean(output)
    loss = mean_sq_undershoot + regularizing_term

    return loss


def regularized_max_loss(output: torch.Tensor, target: torch.Tensor, lambda_reg: float = 1e-6) -> torch.Tensor:
    """
    Variation of the loss function in Forsyth & Li paper:
    A Data Driven Neural Network Approach to Optimal Asset Allocation for
    Target Based Defined Contribution Pension Plans, June 2018
    This function tries to minimize the maximum squared loss (the inf norm of the loss vector)
    in order to attempt to ensure that no scenario ends up with unacceptable loss
    :param output: output wealths
    :param target: target wealths
    :param lambda_reg: regularization constant
    :return: loss
    """
    mean_sq_undershoot = torch.max(torch.clamp(target - output, min=0.0) ** 2)
    regularizing_term = lambda_reg * torch.mean(output)
    loss = mean_sq_undershoot + regularizing_term

    return loss


def max_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Variation of the loss function in Forsyth & Li paper:
    A Data Driven Neural Network Approach to Optimal Asset Allocation for
    Target Based Defined Contribution Pension Plans, June 2018
    This function tries to minimize the maximum squared loss (the inf norm of the loss vector)
    in order to attempt to ensure that no scenario ends up with unacceptable loss
    :param output: output wealths
    :param target: target wealths
    :return: loss
    """
    loss = torch.max(torch.clamp(target - output, min=0.0) ** 2)
    return loss
