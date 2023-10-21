import unittest

from statistics import average, median, sign, variance

class AverageTest(unittest.TestCase):

    def test_should_raise_error_for_empty_set(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            average([])

    def test_should_compute_average(self):
        self.assertAlmostEqual(average([1, 1, 10]), 4)

class VarianceTest(unittest.TestCase):

    def test_should_raise_error_for_empty_set(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            variance([])

    def test_should_compute_variance(self):
        self.assertAlmostEqual(variance([1, 1, 10]), 18)

class MedianTest(unittest.TestCase):

    def test_should_raise_error_for_empty_set(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            median([])

    def test_return_middle_for_odd_number_of_elements(self):
        self.assertAlmostEqual(median([2, 4, 10]), 4)

    def test_return_midpoint_for_even_number_of_elements(self):
        self.assertAlmostEqual(median([2, 4, 9, 10]), 6.5)

    def test_computes_median_of_unsorted_elements(self):
        self.assertAlmostEqual(median([4, 3, 1, 2, 5]), 3)


class SignTest(unittest.TestCase):

    def test_should_return_minus1_for_negative(self):
        self.assertEqual(sign(-123), -1)

    def test_should_return_1_for_positive(self):
        self.assertEqual(sign(123), 1)

    def test_should_return_0_for_0(self):
        self.assertEqual(sign(0), 0)

if __name__ == '__main__':
    unittest.main()
