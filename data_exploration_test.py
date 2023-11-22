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

class CategoricalFieldSummaryTest(unittest.TestCase):

    def test_has_most_frequent_categories(self):
        s = CategoricalFieldSummary(pd.Series(['a', 'b', None, 'a', None]))
        expected = pd.Series({'a':0.4, None:0.4, 'b':0.2})
        self.assertTrue(s.most_frequent.equals(expected))

    def test_has_entropy(self):
        s = CategoricalFieldSummary(pd.Series(['a', 'b', None, 'a', None]))
        self.assertAlmostEqual(s.description['entropy'],
            -2/5*math.log2(2/5) - 2/5*math.log2(2/5) - 1/5*math.log2(1/5))

class NumericalFieldSummaryTest(unittest.TestCase):

    def test_has_description(self):
        s = NumericalFieldSummary(pd.Series([1, 2, 3, 4, None, 6, None]))
        self.assertAlmostEqual(s.description['percent_missing'], 2/7)
        self.assertAlmostEqual(s.description['mean'], 16/5)
        self.assertAlmostEqual(s.description['min'], 1)
        self.assertAlmostEqual(s.description['max'], 6)

    def test_has_entropy_for_low_cardinality_ints(self):
        s = NumericalFieldSummary(pd.Series([1, 2, 2]))
        self.assertAlmostEqual(s.description['entropy'],
            -2/3*math.log2(2/3) - 1/3*math.log2(1/3))

    def test_has_most_frequent_for_low_cardinality_ints(self):
        s = NumericalFieldSummary(pd.Series([1, 2, 2]))
        self.assertIsNotNone(s.most_frequent)

    def test_missing_most_frequent_for_low_cardinality_floats(self):
        s = NumericalFieldSummary(pd.Series([1.1, 2.1, 2.1]))
        self.assertIsNone(s.most_frequent)

class DatasetSummaryTest(unittest.TestCase):

    def test_records_size(self):
        s = DatasetSummary(pd.DataFrame({'a':[1,2,3], 'b':[2,4,6]}))
        self.assertEqual(s.description['num_fields'], 2)
        self.assertEqual(s.description['num_records'], 3)

if __name__ == '__main__':
    unittest.main()
