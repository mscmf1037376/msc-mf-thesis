import numpy as np
import unittest

from wealth_optimizer.performance_stats import plot_trajectory_mean_and_quantiles


class TestPerformanceStatsFunctions(unittest.TestCase):

    def test_plot_mean_wealth(self):
        w1 = np.array([1, 2, 3, 4, 5])
        w2 = 3*np.array([1, 2, 3, 4, 5])

        plot_trajectory_mean_and_quantiles([w1, w2], quantiles=[0.25, 0.75], show=True)


if __name__ == '__main__':
    unittest.main()
