import unittest
import numpy as np
import pandas as pd

from wealth_optimizer.bootstrap import generate_stationary_bootstrap_path


class TestBootstrapMethods(unittest.TestCase):

    def test_stationary_bootstrap(self):
        N = 100
        num_required = 200
        exp_block_size = 15
        historical_data = pd.DataFrame(data={'A': np.random.randn(N), 'B': np.random.randn(N)})

        sampled_data = generate_stationary_bootstrap_path(historical_data, exp_block_size, num_required)
        self.assertEqual(200, sampled_data.shape[0])
        self.assertEqual(2, sampled_data.shape[1])


if __name__ == '__main__':
    unittest.main()
