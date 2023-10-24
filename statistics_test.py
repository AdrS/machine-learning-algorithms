import unittest

from statistics import average, cumulative_sum, median, sign, variance

class AverageTest(unittest.TestCase):

    def test_should_raise_error_for_empty_set(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            average([])

    def test_should_compute_average(self):
        self.assertAlmostEqual(average([1, 1, 10]), 4)

    def test_should_compute_weighted_average(self):
        self.assertAlmostEqual(average([1, 1, 10], [2, 1, 3]),
            (2*1 + 1*1 + 3*10)/(2 + 1 + 3))

class VarianceTest(unittest.TestCase):

    def test_should_raise_error_for_empty_set(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            variance([])

    def test_should_compute_variance(self):
        self.assertAlmostEqual(variance([1, 1, 10]), 18)

class CumulativeSum(unittest.TestCase):

    def test_empty_list(self):
        self.assertEqual(cumulative_sum([]), [])

    def test_sum(self):
        self.assertEqual(cumulative_sum([3, 1, 2, 4]),
            [3, 4, 6, 10])

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

    def test_return_weighted_median(self):
        self.assertAlmostEqual(median([2, 4, 5, 7, 10], [3, 2, 1, 1, 1]), 4)


class SignTest(unittest.TestCase):

    def test_should_return_minus1_for_negative(self):
        self.assertEqual(sign(-123), -1)

    def test_should_return_1_for_positive(self):
        self.assertEqual(sign(123), 1)

    def test_should_return_0_for_0(self):
        self.assertEqual(sign(0), 0)

if __name__ == '__main__':
    unittest.main()
