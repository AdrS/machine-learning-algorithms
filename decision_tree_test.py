import unittest
import decision_tree
import math
from collections import Counter

class EntropyImpurityTest(unittest.TestCase):
    def test_empy_set_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity(Counter()), 0)

    def test_uniform_distribution_has_high_impurity(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity(Counter([1,1,2,2])),
            -math.log(0.5))

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity(Counter([1,2,2, 3,3,3])),
            -math.log(1/6)/6 - math.log(1/3)/3 - math.log(0.5)/2)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity(Counter([1,1,1,1,1,1])),
            0)

class GiniImpurityTest(unittest.TestCase):
    def test_empy_set_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity(Counter()),
            0)

    def test_uniform_distribution_has_high_impurity(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity(Counter([1,1,2,2])),
            0.5)

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity(Counter([1,2,2, 3,3,3])),
            1 - (1/6)**2 - (1/3)**2 - (1/2)**2)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity(Counter([1,1,1,1,1,1])),
            0)

class MeanSquaredErrorImpurityTest(unittest.TestCase):
    def test_empy_set_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.mean_squared_error_impurity([]), 0)

    def test_uniform_distribution(self):
        self.assertAlmostEqual(
            decision_tree.mean_squared_error_impurity([1,1,2,2]), 0.25)

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            decision_tree.mean_squared_error_impurity([1,2,2, 3,3,3]),
            ((1 - 14/6)**2 + 2*(2 - 14/6)**2 + 3*(3 - 14/6)**2)/6)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.mean_squared_error_impurity([1,1,1,1,1,1]),
            0)

class MeanAbsoluteErrorImpurityTest(unittest.TestCase):
    def test_empy_set_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.mean_absolute_error_impurity([]), 0)

    def test_uniform_distribution(self):
        self.assertAlmostEqual(
            decision_tree.mean_absolute_error_impurity([1,2,3,4]),
            (1.5 + 0.5 + 0.5 + 1.5)/4)

    def test_odd_length(self):
        self.assertAlmostEqual(
            decision_tree.mean_absolute_error_impurity([1,2,3]), (1 + 0 + 1)/3)

    def test_even_length(self):
        self.assertAlmostEqual(
            decision_tree.mean_absolute_error_impurity([0, 0, 2, 2]), 1)

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            decision_tree.mean_absolute_error_impurity([1,2,2, 3,3,3]),
            ((14/6 - 1) + 2*(14/6 - 2) + 3*(3 - 14/6))/6)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.mean_absolute_error_impurity([1,1,1,1,1,1]),
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

    def test_to_string(self):
        predicate = decision_tree.ThresholdPredicate(
            feature_index=2, threshold=3)

        self.assertEqual(str(predicate),
            'ThresholdPredicate(feature_index=2, threshold=3)')


class EqualityPredicateTest(unittest.TestCase):
    def test_equality_predicate(self):
        predicate = decision_tree.EqualityPredicate(
            feature_index=1, value='a')

        self.assertTrue(predicate(['a', 'a']))
        self.assertTrue(predicate(['b', 'a']))
        self.assertFalse(predicate(['a', 'b']))
        self.assertFalse(predicate(['b', 'b']))

    def test_to_string(self):
        predicate = decision_tree.EqualityPredicate(
            feature_index=1, value='a')

        self.assertEqual(str(predicate),
            'EqualityPredicate(feature_index=1, value=\'a\')')

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

    def run_fast_slow_subtests(self, subtest):
        'Runs the test with both fast and slow versions of propose splits'
        for slow in [True, False]:
            with self.subTest(slow=slow):
                subtest(slow)

    def subtest_categorical_feature_has_equality_predicates(self, slow):
        X = [['a'], ['b'], ['c']]
        Y = [0, 0, 0]
        impurity = decision_tree.gini_impurity
        predicates = list(
            decision_tree.propose_split_predicates(X, Y, impurity, slow))
        self.assertCountEqual(predicates, [
            decision_tree.EqualityPredicate(0, 'a'),
            decision_tree.EqualityPredicate(0, 'b'),
            decision_tree.EqualityPredicate(0, 'c')
        ])

    def test_categorical_feature_has_equality_predicates(self):
        self.run_fast_slow_subtests(
            self.subtest_categorical_feature_has_equality_predicates)

    def subtest_duplicate_categorical_values(self, slow):
        X = [['a'], ['b'], ['c'], ['b'], ['c'], ['b']]
        Y = [0, 0, 0]
        impurity = decision_tree.gini_impurity
        predicates = list(
            decision_tree.propose_split_predicates(X, Y, impurity, slow))
        self.assertCountEqual(predicates, [
            decision_tree.EqualityPredicate(0, 'a'),
            decision_tree.EqualityPredicate(0, 'b'),
            decision_tree.EqualityPredicate(0, 'c')
        ])

    def test_duplicate_categorical_values(self):
        self.run_fast_slow_subtests(self.subtest_duplicate_categorical_values)

    def subtest_proposes_splits_for_all_features(self, slow):
        X = [['a', 'A'], ['b', 'B']]
        Y = [0, 0, 0]
        impurity = decision_tree.gini_impurity
        predicates = list(
            decision_tree.propose_split_predicates(X, Y, impurity, slow))
        self.assertCountEqual(predicates, [
            decision_tree.EqualityPredicate(0, 'a'),
            decision_tree.EqualityPredicate(0, 'b'),
            decision_tree.EqualityPredicate(1, 'A'),
            decision_tree.EqualityPredicate(1, 'B'),
        ])

    def test_proposes_splits_for_all_features(self):
        self.run_fast_slow_subtests(
            self.subtest_proposes_splits_for_all_features)

        # TODO:
        # numerical features
        # duplicate numerical features
        #...
        # raises error for unsupported feature type
        # find way to run all other tests for slow=true and slow=false

class PartitionTest(unittest.TestCase):
    def test(self):
        pass

class TerminalNodeTest(unittest.TestCase):
    def test_predicts_most_common_value(self):
        node = decision_tree.TerminalNode([1,2,2,2,3,4])
        self.assertEqual(node.predict(['a','b','c']), 2)

    def test_predicts_provided_value(self):
        node = decision_tree.TerminalNode(value=2)
        self.assertEqual(node.predict(['a','b','c']), 2)

    def test_raises_error_if_both_value_and_elements_provided(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            decision_tree.TerminalNode(Y=[1,2,3], value=2)

    def test_to_string(self):
        node = decision_tree.TerminalNode(value=2)
        self.assertEqual(str(node), 'TerminalNode(value=2)')

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

    def test_to_string(self):
        node = decision_tree.InteriorNode(
            predicate=decision_tree.EqualityPredicate(1, 'a'),
            left=decision_tree.InteriorNode(
                predicate=decision_tree.EqualityPredicate(2, 'b'),
                left=decision_tree.TerminalNode(['x']),
                right=decision_tree.TerminalNode(['y'])
            ),
            right=decision_tree.TerminalNode(['z'])
        )
        self.assertEqual(str(node),
'''InteriorNode(predicate=EqualityPredicate(feature_index=1, value='a'),
  left=InteriorNode(predicate=EqualityPredicate(feature_index=2, value='b'),
    left=TerminalNode(value='x'),
    right=TerminalNode(value='y')),
  right=TerminalNode(value='z'))''')

class DecisionTreeClassifierTest(unittest.TestCase):
    def test_stop_splitting_at_max_depth(self):
        model = decision_tree.DecisionTreeClassifier(max_depth=1)
        model.fit([[1], [2], [3], [4]], [0, 1, 0, 1])
        self.assertEqual(type(model.root), decision_tree.InteriorNode)
        self.assertEqual(type(model.root.left), decision_tree.TerminalNode)
        self.assertEqual(type(model.root.right), decision_tree.TerminalNode)

    def test_stop_splitting_pure_nodes(self):
        model = decision_tree.DecisionTreeClassifier()
        model.fit([[1, -1], [0, -2], [1, -3]], [1, 1, 1])
        self.assertEqual(str(model.root),
            str(decision_tree.TerminalNode(value=1)))

    def test_should_split_on_best_feature(self):
        model = decision_tree.DecisionTreeClassifier()
        model.fit([[1, 0], [1, 1], [1, 1]], [0, 1, 1])
        self.assertEqual(
            str(model.root),
            str(decision_tree.InteriorNode(
                predicate=decision_tree.ThresholdPredicate(
                    feature_index=1, threshold=0.5),
                left=decision_tree.TerminalNode(value=0),
                right=decision_tree.TerminalNode(value=1))))

    def test_should_pick_best_split(self):
        model = decision_tree.DecisionTreeClassifier()
        model.fit([[1, -1], [0, -2], [1, -3]], [0, 1, 1])
        self.assertEqual(
            str(model.root),
            str(decision_tree.InteriorNode(
                predicate=decision_tree.ThresholdPredicate(
                    feature_index=1, threshold=-1.5),
                left=decision_tree.TerminalNode(value=1),
                right=decision_tree.TerminalNode(value=0))))

    # TODO: write assertTreesEqual method with nicer error messages
    # TODO: tests for regression

if __name__ == '__main__':
    unittest.main()
