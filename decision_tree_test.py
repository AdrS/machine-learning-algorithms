import unittest
import decision_tree
import math

from decision_tree import partition
from loss import MeanAbsoluteError, MeanSquaredError

class EntropyImpurityTest(unittest.TestCase):

    def test_raises_error_for_empy_distribution(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            decision_tree.entropy_impurity({})

    def test_uniform_distribution_has_high_impurity(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity({1:0.5, 2:0.5}),
            -math.log(0.5))

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity({1:1/6, 2:1/3, 3:1/2}),
            -math.log(1/6)/6 - math.log(1/3)/3 - math.log(0.5)/2)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.entropy_impurity({1:1}), 0)

class GiniImpurityTest(unittest.TestCase):

    def test_raises_error_for_empy_set(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            decision_tree.gini_impurity({})

    def test_uniform_distribution_has_high_impurity(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity({1:0.5, 2:0.5}), 0.5)

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity({1:1/6, 2:1/3, 3:1/2}),
            1 - (1/6)**2 - (1/3)**2 - (1/2)**2)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(
            decision_tree.gini_impurity({1:1}), 0)

class RegressionImpurityTest(unittest.TestCase):

    def __init__(self, loss, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.impurity = decision_tree.get_regression_impurity_fn(loss)

class MeanSquaredErrorImpurityTest(RegressionImpurityTest):

    def __init__(self, *args, **kwargs):
        super().__init__(MeanSquaredError(), *args, **kwargs)

    def test_empy_set_has_0_impurity(self):
        self.assertAlmostEqual(self.impurity([]), 0)

    def test_uniform_distribution(self):
        self.assertAlmostEqual(self.impurity([1,1,2,2]), 0.25)

    def test_unequal_distribution(self):
        self.assertAlmostEqual(self.impurity([1,2,2, 3,3,3]),
            ((1 - 14/6)**2 + 2*(2 - 14/6)**2 + 3*(3 - 14/6)**2)/6)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(self.impurity([1,1,1,1,1,1]), 0)

class MeanAbsoluteErrorImpurityTest(RegressionImpurityTest):

    def __init__(self, *args, **kwargs):
        super().__init__(MeanAbsoluteError(), *args, **kwargs)

    def test_empy_set_has_0_impurity(self):
        self.assertAlmostEqual(self.impurity([]), 0)

    def test_uniform_distribution(self):
        self.assertAlmostEqual(self.impurity([1,2,3,4]),
            (1.5 + 0.5 + 0.5 + 1.5)/4)

    def test_odd_length(self):
        self.assertAlmostEqual(self.impurity([1,2,3]), (1 + 0 + 1)/3)

    def test_even_length(self):
        self.assertAlmostEqual(self.impurity([0, 0, 2, 2]), 1)

    def test_unequal_distribution(self):
        self.assertAlmostEqual(
            self.impurity([1,2,2, 3,3,3]),
            ((14/6 - 1) + 2*(14/6 - 2) + 3*(3 - 14/6))/6)

    def test_point_distribution_has_0_impurity(self):
        self.assertAlmostEqual(self.impurity([1,1,1,1,1,1]), 0)

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

    def test_predicates_on_different_fields_are_not_equal(self):
        a = decision_tree.ThresholdPredicate(
            feature_index=1, threshold=3)
        b = decision_tree.ThresholdPredicate(
            feature_index=2, threshold=3)
        self.assertNotEqual(a, b)

    def test_predicates_with_different_thresholds_are_not_equal(self):
        a = decision_tree.ThresholdPredicate(
            feature_index=1, threshold=3)
        b = decision_tree.ThresholdPredicate(
            feature_index=1, threshold=4)
        self.assertNotEqual(a, b)

    def test_equal_predicates(self):
        a = decision_tree.ThresholdPredicate(
            feature_index=1, threshold=3)
        b = decision_tree.ThresholdPredicate(
            feature_index=1, threshold=3)
        self.assertEqual(a, b)


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

    def test_predicates_on_different_fields_are_not_equal(self):
        a = decision_tree.EqualityPredicate(
            feature_index=1, value='a')
        b = decision_tree.EqualityPredicate(
            feature_index=2, value='a')
        self.assertNotEqual(a, b)

    def test_predicates_with_different_values_are_not_equal(self):
        a = decision_tree.EqualityPredicate(
            feature_index=1, value='a')
        b = decision_tree.EqualityPredicate(
            feature_index=1, value='b')
        self.assertNotEqual(a, b)

    def test_equal_predicates(self):
        a = decision_tree.EqualityPredicate(
            feature_index=1, value='a')
        b = decision_tree.EqualityPredicate(
            feature_index=1, value='a')
        self.assertEqual(a, b)

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

    def test_is_numerical_false_for_list(self):
        self.assertFalse(decision_tree.is_numerical([]))

    def test_is_numerical_false_for_dictionary(self):
        self.assertFalse(decision_tree.is_numerical({}))

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

    def test_is_categorical_false_for_list(self):
        self.assertFalse(decision_tree.is_categorical([]))

    def test_is_categorical_false_for_dictionary(self):
        self.assertFalse(decision_tree.is_categorical({}))

class TerminalNodeTest(unittest.TestCase):

    def test_predicts_provided_value(self):
        node = decision_tree.TerminalNode(2)
        self.assertEqual(node.predict(['a','b','c']), 2)

    def test_predict_prob_return_distribution(self):
        node = decision_tree.TerminalNode(2, {1:0.1, 2:0.8, 3:0.1})
        self.assertEqual(node.predict_prob(['a','b','c']),
            {1:0.1, 2:0.8, 3:0.1})

    def test_predict_prob_raises_error_if_not_distribution(self):
        node = decision_tree.TerminalNode(2)
        with self.assertRaisesRegex(TypeError, 'distribution'):
            node.predict_prob(['a','b','c'])

    def test_to_string(self):
        node = decision_tree.TerminalNode(2)
        self.assertEqual(str(node), 'TerminalNode(value=2)')

class InteriorNodeTest(unittest.TestCase):
    def test_predicts_left_if_predicate_true(self):
        node = decision_tree.InteriorNode(
            predicate=decision_tree.EqualityPredicate(1, 'a'),
            left=decision_tree.TerminalNode('left'),
            right=decision_tree.TerminalNode('right')
        )
        self.assertEqual(node.predict(['x','a','x']), 'left')

    def test_predicts_right_if_predicate_false(self):
        node = decision_tree.InteriorNode(
            predicate=decision_tree.EqualityPredicate(1, 'a'),
            left=decision_tree.TerminalNode('left'),
            right=decision_tree.TerminalNode('right')
        )
        self.assertEqual(node.predict(['x','b','x']), 'right')

    def test_to_string(self):
        node = decision_tree.InteriorNode(
            predicate=decision_tree.EqualityPredicate(1, 'a'),
            left=decision_tree.InteriorNode(
                predicate=decision_tree.EqualityPredicate(2, 'b'),
                left=decision_tree.TerminalNode('x'),
                right=decision_tree.TerminalNode('y')
            ),
            right=decision_tree.TerminalNode('z')
        )
        self.assertEqual(str(node),
'''InteriorNode(predicate=EqualityPredicate(feature_index=1, value='a'),
  left=InteriorNode(predicate=EqualityPredicate(feature_index=2, value='b'),
    left=TerminalNode(value='x'),
    right=TerminalNode(value='y')),
  right=TerminalNode(value='z'))''')


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

    def subtest_proposes_midpoint_between_values(self, slow):
        X = [[2], [3]]
        Y = [0, 1]
        impurity = decision_tree.gini_impurity
        predicates = list(
            decision_tree.propose_split_predicates(X, Y, impurity, slow))
        self.assertCountEqual(predicates, [
            decision_tree.ThresholdPredicate(0, 2.5)
        ])

    def test_proposes_midpoint_between_values(self):
        self.run_fast_slow_subtests(
            self.subtest_proposes_midpoint_between_values)

    def test_slow_prosposes_all_unique_splits(self):
        X = [[2], [2], [3], [4]]
        Y = [0, 0, 0, 0]
        impurity = decision_tree.gini_impurity
        predicates = list(
            decision_tree.propose_split_predicates(X, Y, impurity, slow=True))
        self.assertCountEqual(predicates, [
            decision_tree.ThresholdPredicate(0, 2.5),
            decision_tree.ThresholdPredicate(0, 3.5)
        ])

    def test_fast_prosposes_best_split_for_feature(self):
        X = [[2], [2], [3], [4]]
        Y = [0, 0, 1, 1]
        impurity = decision_tree.gini_impurity
        predicates = list(
            decision_tree.propose_split_predicates(X, Y, impurity, slow=False))
        self.assertCountEqual(predicates, [
            decision_tree.ThresholdPredicate(0, 2.5)
        ])

    def subtest_raises_error_for_unsupported_feature_type(self, slow):
        X = [[{}]]
        Y = [0]
        impurity = decision_tree.gini_impurity
        with self.assertRaisesRegex(TypeError, 'numerical or categorical'):
            list(decision_tree.propose_split_predicates(X, Y, impurity, slow))

    def test_raises_error_for_unsupported_feature_type(self):
        self.run_fast_slow_subtests(
            self.subtest_raises_error_for_unsupported_feature_type)

# TODO: Find a way to run all other tests for slow=True and slow=False

class PartitionTest(unittest.TestCase):

    def test_partition_without_weights(self):
        X = [[4], [3], [1], [2]]
        Y = [[0], [1], [1], [0]]
        W = None
        predicate = decision_tree.ThresholdPredicate(
            feature_index=0, threshold=2.5)

        X_left, Y_left, X_right, Y_right, W_left, W_right = partition(
            predicate, X, Y, W)
        self.assertEqual(X_left, [[1], [2]])
        self.assertEqual(Y_left, [[1], [0]])
        self.assertEqual(X_right, [[4], [3]])
        self.assertEqual(Y_right, [[0], [1]])
        self.assertIsNone(W_left)
        self.assertIsNone(W_right)

    def test_partition_with_weights(self):
        X = [[4], [3], [1], [2]]
        Y = [[0], [1], [1], [0]]
        W = [1, 2, 3, 4]
        predicate = decision_tree.ThresholdPredicate(
            feature_index=0, threshold=2.5)

        X_left, Y_left, X_right, Y_right, W_left, W_right = partition(
            predicate, X, Y, W)
        self.assertEqual(X_left, [[1], [2]])
        self.assertEqual(Y_left, [[1], [0]])
        self.assertEqual(X_right, [[4], [3]])
        self.assertEqual(Y_right, [[0], [1]])
        self.assertEqual(W_left, [3, 4])
        self.assertEqual(W_right, [1, 2])

class DecisionTreeTestCase(unittest.TestCase):

    def assertTreesEqual(self, lhs_root, rhs_root):
        self.assertEqual(str(lhs_root), str(rhs_root))

class DecisionTreeClassifierTest(DecisionTreeTestCase):

    def test_stop_splitting_at_max_depth(self):
        model = decision_tree.DecisionTreeClassifier(max_depth=1)
        model.fit([[1], [2], [3], [4]], [0, 1, 0, 1])
        self.assertEqual(type(model.root), decision_tree.InteriorNode)
        self.assertEqual(type(model.root.left), decision_tree.TerminalNode)
        self.assertEqual(type(model.root.right), decision_tree.TerminalNode)

    def test_stop_splitting_pure_nodes(self):
        model = decision_tree.DecisionTreeClassifier()
        model.fit([[1, -1], [0, -2], [1, -3]], [1, 1, 1])
        self.assertTreesEqual(model.root, decision_tree.TerminalNode(1))

    def test_should_split_on_best_feature(self):
        model = decision_tree.DecisionTreeClassifier()
        model.fit([[1, 0], [1, 1], [1, 1]], [0, 1, 1])
        self.assertTreesEqual(
            model.root,
            decision_tree.InteriorNode(
                predicate=decision_tree.ThresholdPredicate(
                    feature_index=1, threshold=0.5),
                left=decision_tree.TerminalNode(0),
                right=decision_tree.TerminalNode(1)))

    def test_stop_for_trivial_splits(self):
        model = decision_tree.DecisionTreeClassifier()
        # Note: this is not pure because the targets are different. There are
        # no possible splits except trivial splits that are empty on one side
        # because all the features are the same.
        model.fit([[1], [1], [1]], [1, 0, 1])
        self.assertTreesEqual(model.root, decision_tree.TerminalNode(1))

    def test_should_pick_best_split(self):
        model = decision_tree.DecisionTreeClassifier()
        model.fit([[1, -1], [0, -2], [1, -3]], [0, 1, 1])
        self.assertTreesEqual(
            model.root,
            decision_tree.InteriorNode(
                predicate=decision_tree.ThresholdPredicate(
                    feature_index=1, threshold=-1.5),
                left=decision_tree.TerminalNode(1),
                right=decision_tree.TerminalNode(0)))

    def test_should_pick_best_split_from_weighted_data(self):
        model = decision_tree.DecisionTreeClassifier(max_depth=1)
        model.fit(
            X=[[1], [2], [3], [4], [5], [6], [7]],
            Y=[0, 0, 1, 0, 1, 1, 1],
            weights=[10, 1, 1, 1, 1, 1, 1])
        # Without weights the threshold is 4.5
        self.assertTreesEqual(
            model.root,
            decision_tree.InteriorNode(
                predicate=decision_tree.ThresholdPredicate(
                    feature_index=0, threshold=2.5),
                left=decision_tree.TerminalNode(0),
                right=decision_tree.TerminalNode(1)))

    def test_predict(self):
        model = decision_tree.DecisionTreeClassifier()
        # y = (x1 > 0) xor (x2 > 0)
        model.fit([[-1, -1], [1, -1], [-1, 1], [1, 1]], [0, 1, 1, 0])

        self.assertEqual(model.predict([
                [-2, -2], [2, -2], [-2, 2], [2, 2],
                [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]
            ]),
            [0, 1, 1, 0, 0, 1, 1, 0])

    def test_predict_prob(self):
        model = decision_tree.DecisionTreeClassifier()
        model.fit([[1], [1], [1], [2], [2]], [0, 1, 1, 0, 0])

        self.assertEqual(model.predict_prob([
                [0], [1], [2], [3],
            ]), [
                {0:1/3, 1:2/3},
                {0:1/3, 1:2/3},
                {0:1},
                {0:1}
            ])

class DecisionTreeRegressorTest(DecisionTreeTestCase):

    def test_stop_splitting_pure_nodes(self):
        model = decision_tree.DecisionTreeRegressor()
        model.fit([[1, -1], [0, -2], [1, -3]], [3, 3, 3])
        self.assertTreesEqual(model.root, decision_tree.TerminalNode(3.0))

    def test_should_split_on_best_feature(self):
        model = decision_tree.DecisionTreeRegressor()
        model.fit([[1, 0], [1, 1], [1, 1]], [3, 5, 6])
        self.assertTreesEqual(
            model.root,
            decision_tree.InteriorNode(
                predicate=decision_tree.ThresholdPredicate(
                    feature_index=1, threshold=0.5),
                left=decision_tree.TerminalNode(3.0),
                right=decision_tree.TerminalNode(5.5)))

    def test_should_pick_best_split(self):
        model = decision_tree.DecisionTreeRegressor()
        model.fit([[1, -1], [0, -2], [1, -3]], [3, 5, 5])
        self.assertTreesEqual(
            model.root,
            decision_tree.InteriorNode(
                predicate=decision_tree.ThresholdPredicate(
                    feature_index=1, threshold=-1.5),
                left=decision_tree.TerminalNode(5.0),
                right=decision_tree.TerminalNode(3.0)))

    def test_should_pick_best_split_from_weighted_data(self):
        model = decision_tree.DecisionTreeClassifier(max_depth=1)
        model.fit(
            X=[[1], [2], [3], [4], [5], [6], [7]],
            Y=[3, 3, 5, 3, 5, 5, 5],
            weights=[10, 1, 1, 1, 1, 1, 1])
        # Without weights the threshold is 4.5
        self.assertTreesEqual(
            model.root,
            decision_tree.InteriorNode(
                predicate=decision_tree.ThresholdPredicate(
                    feature_index=0, threshold=2.5),
                left=decision_tree.TerminalNode(3),
                right=decision_tree.TerminalNode(5)))

    def test_predict(self):
        model = decision_tree.DecisionTreeRegressor()
        model.fit([[1, -1], [0, -2], [1, -3]], [3, 5, 5])

        self.assertEqual(model.predict([[1, -1], [0, -2], [1, -3]]), [3, 5, 5])

    # TODO: prune, score, loss, export text
    # TODO: information_gain, feature importance

class DecisionStumpClassifierTest(DecisionTreeTestCase):

    def test_stump_has_single_split(self):
        model = decision_tree.DecisionStumpClassifier()
        model.fit([[1], [2], [3]], [0, 1, 0])
        self.assertEqual(type(model.root), decision_tree.InteriorNode)
        self.assertEqual(type(model.root.left), decision_tree.TerminalNode)
        self.assertEqual(type(model.root.right), decision_tree.TerminalNode)

class DecisionStumpRegressorTest(DecisionTreeTestCase):

    def test_stump_has_single_split(self):
        model = decision_tree.DecisionStumpRegressor()
        model.fit([[1], [2], [3]], [0, 4, 5])
        self.assertEqual(type(model.root), decision_tree.InteriorNode)
        self.assertEqual(type(model.root.left), decision_tree.TerminalNode)
        self.assertEqual(type(model.root.right), decision_tree.TerminalNode)

if __name__ == '__main__':
    unittest.main()
