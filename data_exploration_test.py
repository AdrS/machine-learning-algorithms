import unittest
import math
import numpy as np
import pandas as pd

from data_exploration import *

class FieldSummaryTest(unittest.TestCase):

    def test_records_missing_values(self):
        s = FieldSummary(pd.Series([1, 2, None, 4, None]))
        self.assertEqual(s.description['percent_missing'], 0.4)

    def test_records_description(self):
        s = FieldSummary(pd.Series([1, 2, None, 4, None]))
        self.assertEqual(s.description['max'], 4)

class IsNumericalTest(unittest.TestCase):

    def test_float_is_numerical(self):
        series = pd.Series([1.2, 3.4])
        self.assertTrue(is_numerical(series))

    def test_int_is_numerical(self):
        series = pd.Series([1, 2, 4])
        self.assertTrue(is_numerical(series))

    def test_bool_is_numerical(self):
        series = pd.Series([True, False])
        self.assertTrue(is_numerical(series))

    def test_string_not_numerical(self):
        series = pd.Series(['hi', 'bye'])
        self.assertFalse(is_numerical(series))

    def test_missing_values_ok(self):
        series = pd.Series([1, 2, None, 4, None])
        self.assertTrue(is_numerical(series))

class CategoricalFieldSummaryTest(unittest.TestCase):

    def test_has_most_frequent_categories(self):
        s = CategoricalFieldSummary(pd.Series(['a', 'b', None, 'a', None]))
        expected = pd.Series({'a':0.4, None:0.4, 'b':0.2})
        self.assertTrue(s.most_frequent.equals(expected))

    def test_has_entropy(self):
        s = CategoricalFieldSummary(pd.Series(['a', 'b', None, 'a', None]))
        self.assertAlmostEqual(s.description['entropy'],
            -2/5*math.log2(2/5) - 2/5*math.log2(2/5) - 1/5*math.log2(1/5))

class DatasetSummaryTest(unittest.TestCase):

    def test_records_size(self):
        s = DatasetSummary(pd.DataFrame({'a':[1,2,3], 'b':[2,4,6]}))
        self.assertEqual(s.description['num_fields'], 2)
        self.assertEqual(s.description['num_records'], 3)

if __name__ == '__main__':
    unittest.main()
