import unittest

from loss import HuberLossFn, MeanAbsoluteError, MeanSquaredError

class MeanSquaredErrorTest(unittest.TestCase):

    def test_should_use_mean_as_prior(self):
        loss = MeanSquaredError()
        self.assertAlmostEqual(loss.prior([1, 4, 7]), 4)

    def test_loss(self):
        loss = MeanSquaredError()
        targets = [1, 4, 7]
        predictions = [1, 3, 10]
        expected = (0 + 1**2 + 3**2)/3
        self.assertAlmostEqual(loss.loss(targets, predictions), expected)

    def test_gradients(self):
        loss = MeanSquaredError()
        targets = [1, 4, 7]
        predictions = [1, 3, 10]
        # L = (t - p)*2
        # -dL/dp = 2(t - p)
        expected = [0, 2*(4 - 3), 2*(7 - 10)]
        self.assertEqual(
            loss.negative_gradients(targets, predictions), expected)

class MeanAbsoluteErrorTest(unittest.TestCase):

    def test_should_use_media_as_prior(self):
        loss = MeanAbsoluteError()
        self.assertAlmostEqual(loss.prior([1, 3, 4, 7, 100]), 4)

    def test_loss(self):
        loss = MeanAbsoluteError()
        targets = [1, 4, 7]
        predictions = [1, 3, 10]
        expected = (0 + 1 + 3)/3
        self.assertAlmostEqual(loss.loss(targets, predictions), expected)

    def test_gradients(self):
        loss = MeanAbsoluteError()
        targets = [1, 4, 7]
        predictions = [1, 3, 10]
        # L = |t - p|
        # - dL/dp = 1 if t > p
        #           -1 if t < p
        expected = [0, 1, -1]
        self.assertEqual(
            loss.negative_gradients(targets, predictions), expected)

class HuberLossFnTest(unittest.TestCase):

    def test_should_use_media_as_prior(self):
        loss = HuberLossFn()
        self.assertAlmostEqual(loss.prior([1, 3, 4, 7, 100]), 4)

    def test_loss_zero(self):
        loss = HuberLossFn(delta=1)
        self.assertAlmostEqual(loss.loss([1], [1]), 0)

    def test_loss_less_than_delta(self):
        loss = HuberLossFn(delta=2)
        self.assertAlmostEqual(loss.loss([1], [1.6]), 0.5*0.6*0.6)

    def test_loss_equals_delta(self):
        loss = HuberLossFn(delta=2)
        self.assertAlmostEqual(loss.loss([1], [3]), 0.5*2*2)
        self.assertAlmostEqual(loss.loss([1], [3]), 2*(2 - 0.5*2))

    def test_loss_greater_than_delta(self):
        loss = HuberLossFn(delta=2)
        self.assertAlmostEqual(loss.loss([1], [6]), 2*(5 - 0.5*2))

    def test_gradients(self):
        loss = HuberLossFn(delta=1)
        targets = [1, 2.5, 4, 7]
        predictions = [1, 2, 3, 10]
        expected = [0, 0.5, 1, -1]
        self.assertEqual(
            loss.negative_gradients(targets, predictions), expected)

if __name__ == '__main__':
    unittest.main()
