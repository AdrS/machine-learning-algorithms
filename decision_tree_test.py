import unittest
import decision_tree
import math

class EntropyImpurityTest(unittest.TestCase):
    def test_empy_set_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity([]), 0)

    def test_uniform_distribution_has_high_impurity(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity([1,1,2,2]),
            -math.log(0.5))

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity([1,2,2, 3,3,3]),
            -math.log(1/6)/6 - math.log(1/3)/3 - math.log(0.5)/2)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity([1,1,1,1,1,1]),
            0)

class GiniImpurityTest(unittest.TestCase):
    def test_empy_set_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity([]),
            0)

    def test_uniform_distribution_has_high_impurity(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity([1,1,2,2]),
            0.5)

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity([1,2,2, 3,3,3]),
            1 - (1/6)**2 - (1/3)**2 - (1/2)**2)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity([1,1,1,1,1,1]),
            0)

class ThresholdPredicateTest(unittest.TestCase):

    def test_threshold_predicate(self):
        predicate = decision_tree.ThresholdPredicate(
            feature_index=0, threshold=3)

        self.assertTrue(predicate([2]))
        self.assertFalse(predicate([3]))
        self.assertFalse(predicate([4]))

        predicate = decision_tree.ThresholdPredicate(
            feature_index=2, threshold=3)
        self.assertFalse(predicate([2, 2, 3, 2]))
        self.assertTrue(predicate([4, 4, 2, 4]))

class EqualityPredicateTest(unittest.TestCase):
    def test_equality_predicate(self):
        predicate = decision_tree.EqualityPredicate(
            feature_index=1, value='a')

        self.assertTrue(predicate(['a', 'a']))
        self.assertTrue(predicate(['b', 'a']))
        self.assertFalse(predicate(['a', 'b']))
        self.assertFalse(predicate(['b', 'b']))

class IsNumericalTest(unittest.TestCase):
    def test_is_numerical_true_for_int(self):
        self.assertTrue(decision_tree.is_numerical(123))

    def test_is_numerical_true_for_float(self):
        self.assertTrue(decision_tree.is_numerical(1.23))

    def test_is_numerical_false_for_string(self):
        self.assertFalse(decision_tree.is_numerical('abc'))

    def test_is_numerical_false_for_object(self):
        obj = decision_tree.EqualityPredicate(1,'123')
        self.assertFalse(decision_tree.is_numerical(obj))

class IsCategoricalTest(unittest.TestCase):
    def test_is_categorical_false_for_int(self):
        self.assertFalse(decision_tree.is_categorical(123))

    def test_is_categorical_false_for_float(self):
        self.assertFalse(decision_tree.is_categorical(1.23))

    def test_is_categorical_true_for_string(self):
        self.assertTrue(decision_tree.is_categorical('abc'))

    def test_is_categorical_false_for_object(self):
        obj = decision_tree.EqualityPredicate(1,'123')
        self.assertFalse(decision_tree.is_categorical(obj))

class ProposeSplitPartitionsTest(unittest.TestCase):

    def test_categorical_feature_has_equality_predicates(self):
        X = [['a'], ['b'], ['c']]
        predicates = list(decision_tree.propose_split_predicates(X))
        self.assertCountEqual(predicates, [
            decision_tree.EqualityPredicate(0, 'a'),
            decision_tree.EqualityPredicate(0, 'b'),
            decision_tree.EqualityPredicate(0, 'c')
        ])

    def test_duplicate_categorical_values(self):
        X = [['a'], ['b'], ['c'], ['b'], ['c'], ['b']]
        predicates = list(decision_tree.propose_split_predicates(X))
        self.assertCountEqual(predicates, [
            decision_tree.EqualityPredicate(0, 'a'),
            decision_tree.EqualityPredicate(0, 'b'),
            decision_tree.EqualityPredicate(0, 'c')
        ])

    def test_proposes_splits_for_all_features(self):
        X = [['a', 'A'], ['b', 'B']]
        predicates = list(decision_tree.propose_split_predicates(X))
        self.assertCountEqual(predicates, [
            decision_tree.EqualityPredicate(0, 'a'),
            decision_tree.EqualityPredicate(0, 'b'),
            decision_tree.EqualityPredicate(1, 'A'),
            decision_tree.EqualityPredicate(1, 'B'),
        ])

        # TODO: multiple indicies
        # numerical features
        # duplicate numerical features
        #...
        # raises error for unsupported feature type
        pass

class PartitionTest(unittest.TestCase):
    def test(self):
        pass

# TODO: stop splitting test

class TerminalNodeTest(unittest.TestCase):
    def test_predicts_most_common_value(self):
        node = decision_tree.TerminalNode([1,2,2,2,3,4])
        self.assertEqual(node.predict(['a','b','c']), 2)

class InteriorNodeTest(unittest.TestCase):
    def test_predicts_left_if_predicate_true(self):
        node = decision_tree.InteriorNode(
            predicate=decision_tree.EqualityPredicate(1, 'a'),
            left=decision_tree.TerminalNode(['left']),
            right=decision_tree.TerminalNode(['right'])
        )
        self.assertEqual(node.predict(['x','a','x']), 'left')

    def test_predicts_right_if_predicate_false(self):
        node = decision_tree.InteriorNode(
            predicate=decision_tree.EqualityPredicate(1, 'a'),
            left=decision_tree.TerminalNode(['left']),
            right=decision_tree.TerminalNode(['right'])
        )
        self.assertEqual(node.predict(['x','b','x']), 'right')

# TODO: construct decision tree test

if __name__ == '__main__':
    unittest.main()
