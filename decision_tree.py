import math

from collections import Counter

class TerminalNode:
    def __init__(self, Y):
        # Predict the most common class
        self.value = Counter(Y).most_common(1)[0][0]

    def predict(self, x):
        return self.value

class InteriorNode:
    def __init__(self, predicate, left, right):
        self.predicate = predicate
        self.left = left
        self.right = right

    def predict(self, x):
        if self.predicate(x):
            return self.left.predict(x)
        else:
            return self.right.predict(x)

def is_numerical(x):
    return type(x) == int or type(x) == float

def is_categorical(x):
    return type(x) == str

def entropy_impurity(Y):
    counts = Counter(Y)
    entropy = 0
    for value_count in counts.values():
        p = value_count/len(Y)
        entropy -= p*math.log(p)
    return entropy

def gini_impurity(Y):
    if not Y:
        return 0
    counts = Counter(Y)
    sum_squares = 0
    for value_count in counts.values():
        p = value_count/len(Y)
        sum_squares += p*p
    return 1 - sum_squares

# TODO: log loss impurity

class ThresholdPredicate:
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold

    def __call__(self, x):
        return x[self.feature_index] < self.threshold

    def __repr__(self):
        return 'ThresholdPredicate(feature_index=%d, threshold=%r)' % (
            self.feature_index, self.threshold)

    def __eq__(self, other):
        return (type(other) == ThresholdPredicate and
            self.feature_index == other.feature_index and
            self.threshold == other.threshold)

class EqualityPredicate:
    def __init__(self, feature_index, value):
        self.feature_index = feature_index
        self.value = value

    def __call__(self, x):
        return x[self.feature_index] == self.value

    def __repr__(self):
        return 'EqualityPredicate(feature_index=%d, value=%r)' % (
            self.feature_index, self.value)

    def __eq__(self, other):
        return (type(other) == EqualityPredicate and
            self.feature_index == other.feature_index and
            self.value == other.value)

def propose_split_predicates(X):
    for i in range(len(X[0])):
        unique_values = {x[i] for x in X}
        if is_numerical(X[0][i]):
            ordered_values = sorted(unique_values)
            for l, r in zip(ordered_values, ordered_values[1:]):
                midpoint = (l + r)/2.0
                yield ThresholdPredicate(i, midpoint)
        elif is_categorical(X[0][i]):
            for value in unique_values:
                yield EqualityPredicate(i, value)
        else:
            raise TypeError('Feature must be numerical or categorical')

def partition(X, Y, split_predicate):
    X_left, Y_left, X_right, Y_right = [], [], [], []
    for x, y in zip(X, Y):
        if split_predicate(x):
            X_left.append(x)
            Y_left.append(y)
        else:
            X_right.append(x)
            Y_right.append(y)
    return X_left, Y_left, X_right, Y_right

def is_pure(Y):
    for y in Y:
        if y != Y[0]:
            return False
    return True

class DecisionTreeClassifier:
    def __init__(self, impurity=gini_impurity, max_depth=99999):
        self.root = None
        self.impurity = impurity
        self.max_depth = max_depth

    def construct_decision_tree(self, X, Y, depth):
        if is_pure(Y) or depth >= self.max_depth:
            return TerminalNode(Y)

        best_score = -1
        best_split = None
        for split_predicate in propose_split_predicates(X):
            X_left, Y_left, X_right, Y_right = partition(X, Y, split_predicate)
            if not X_left or not X_right:
                continue
            # IG = I(Y) - p_left*I(Y_left) - p_right*I(Y_right)
            score = (self.impurity(Y_left)*len(Y_left) +
                        self.impurity(Y_right)*len(Y_right))
            if score > best_score:
                best_score = score
                best_split = (split_predicate, X_left, Y_left, X_right, Y_right)

        if not best_split:
            # TODO: test additional stopping criteria
            # - min samples per leaf
            # - min information gain
            # TODO: best first search for to support max leaf nodes
            return TerminalNode(Y)

        split_predicate, X_left, Y_left, X_right, Y_right = best_split
        return InteriorNode(
            split_predicate,
            self.construct_decision_tree(X_left, Y_left, depth + 1),
            self.construct_decision_tree(X_right, Y_right, depth + 1))


    def fit(self, X, Y):
        self.root = self.construct_decision_tree(X, Y, depth=0)

    def predict(self, X):
        return [self.root.predict(x) for x in X]

    def score(self, X, Y):
        num_correct = 0
        for prediction, target in zip(self.predict(X), Y):
            num_correct += (prediction == target)
        return num_correct/len(X)

    def export_text(self, feature_names):
        parts = []
        def append_predicate(predicate, is_true):
            parts.append(feature_names[predicate.feature_index])
            if type(predicate) == ThresholdPredicate:
                if is_true:
                    parts.append(' <= ')
                else:
                    parts.append(' > ')
                parts.append(str(predicate.threshold))
            elif type(predicate) == EqualityPredicate:
                if is_true:
                    parts.append(' = ')
                else:
                    parts.append(' != ')
                parts.append(str(predicate.value))
            else:
                raise TypeError(f'Unknown predicate type {type(predicate)}')
            parts.append('\n')

        def f(node, indent):
            if type(node) == TerminalNode:
                parts.append(' '*indent)
                parts.append(f'Value: {node.value}\n')
            elif type(node) == InteriorNode:
                parts.append(' '*indent)
                append_predicate(node.predicate, is_true=True)
                f(node.left, indent + 2)
                parts.append(' '*indent)
                append_predicate(node.predicate, is_true=False)
                f(node.right, indent + 2)
            else:
                raise TypeError(f'Unknown node type {type(node)}')
        f(self.root, indent=0)
        return ''.join(parts)
