"""Independent numerical tests; run with python -m unittest discover -s lab02/src."""
import unittest
import numpy as np
from network import initialize, forward, backward, cross_entropy


class NetworkTests(unittest.TestCase):
    def test_stable_loss_at_large_logits(self):
        logits = np.array([[10000., 10001., 9999.], [-10000., -10001., -9999.]])
        targets = np.eye(3)[[1, 2]]
        loss = cross_entropy(logits, targets)
        self.assertAlmostEqual(loss, np.log(1 + np.exp(-1) + np.exp(-2)), places=12)
        self.assertAlmostEqual(loss, cross_entropy(logits + 12345, targets), places=12)

    def test_all_six_gradient_arrays(self):
        rng = np.random.RandomState(42)
        parameters = initialize(4)
        for name in ("W1", "W2", "W3"):
            parameters[name] *= 10
        parameters["b1"][:] = 1
        parameters["b2"][:] = 1
        X = rng.uniform(0, 1, (5, 64))
        Y = np.eye(10)[[0, 2, 4, 6, 8]]
        probabilities, cache = forward(X, parameters)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-14)
        gradients = backward(parameters, cache, Y)
        for name, parameter in parameters.items():
            with self.subTest(parameter=name):
                self.assertEqual(parameter.shape, gradients[name].shape)
                direction = rng.normal(size=parameter.shape)
                direction /= np.linalg.norm(direction)
                original = parameter.copy()
                parameter[:] = original + 1e-5 * direction
                _, plus = forward(X, parameters)
                parameter[:] = original - 1e-5 * direction
                _, minus = forward(X, parameters)
                parameter[:] = original
                for index in (1, 3):
                    np.testing.assert_array_equal(plus[index] > 0, minus[index] > 0)
                numerical = (cross_entropy(plus[5], Y) - cross_entropy(minus[5], Y)) / 2e-5
                analytical = np.sum(direction * gradients[name])
                np.testing.assert_allclose(analytical, numerical, atol=1e-9, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
