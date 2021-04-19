import unittest

import torch

from wealth_optimizer.loss_functions import forsyth_li_regularized_loss, regularized_max_loss


class TestLossFunctions(unittest.TestCase):

    def test_regularized_loss(self):
        w_out = torch.tensor(1.0)
        w_target = torch.tensor(1.0)

        loss = forsyth_li_regularized_loss(w_out, w_target).numpy()
        self.assertAlmostEqual(1e-6, loss)

        loss = forsyth_li_regularized_loss(w_out, w_target, 1e-2).numpy()
        self.assertAlmostEqual(1e-2, loss)

    def test_regularized_max_loss(self):
        w_out = torch.tensor(1.0)
        w_target = torch.tensor(1.0)

        loss = regularized_max_loss(w_out, w_target).numpy()
        self.assertAlmostEqual(1e-6, loss)

        loss = regularized_max_loss(w_out, w_target, 1e-2).numpy()
        self.assertAlmostEqual(1e-2, loss)

        w_out = torch.tensor(0.9)
        w_target = torch.tensor(1.0)
        loss = regularized_max_loss(w_out, w_target, 1e-6).numpy()
        self.assertAlmostEqual(1e-6 * w_out.numpy() + 0.1 * 0.1, loss)


if __name__ == '__main__':
    unittest.main()
