import math

from collections import Counter
from loss import MeanSquaredError
from statistics import average, variance, median

def entropy_impurity(counts):
    if not counts:
        raise ValueError('Input must be non-empty')
    entropy = 0
    total = sum(counts.values())
    for value_count in counts.values():
        p = value_count/total
        entropy -= p*math.log(p)
    return entropy

def gini_impurity(counts):
    if not counts:
        raise ValueError('Input must be non-empty')
    total = sum(counts.values())
    sum_squares = 0
    for value_count in counts.values():
        p = value_count/total
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

def is_numerical(x):
    return type(x) == int or type(x) == float

def is_categorical(x):
    return type(x) == str

class TerminalNode:
    def __init__(self, value, distribution=None):
        self.value = value
        self.distribution = distribution

    def predict(self, x):
        return self.value

    def predict_prob(self, x):
        if not self.distribution:
            raise TypeError('TerminalNode does not have a distribution')
        return self.distribution

    def __repr__(self, indent=None):
        return 'TerminalNode(value=%r)' % (self.value,)

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

    def predict_prob(self, x):
        if self.predicate(x):
            return self.left.predict_prob(x)
        else:
            return self.right.predict_prob(x)

    def __repr__(self, indent=0):
        return 'InteriorNode(predicate=%r,\n%sleft=%s,\n%sright=%s)' % (
            self.predicate,
            ' '*(indent + 2),
            self.left.__repr__(indent + 2),
            ' '*(indent + 2),
            self.right.__repr__(indent + 2))

def compute_probability_distribution(X, weights=None):
    '''Returns the probability distribution for the frequency of values in X.

    X - list of values
    weights - non-negative weights to weight elements of X differently

    Returns: a map from unique values in X to the probability
    '''
    if weights:
        P = {}
        total = 0
        for x, w in zip(X, weights):
            if x in P:
                P[x] += w
            else:
                P[x] = w
            total += w
        for x in P:
            P[x] /= total
    else:
        P = {}
        for x in X:
            if x in P:
                P[x] += 1
            else:
                P[x] = 1
        for x in P:
            P[x] /= len(X)
    return P

def fast_propose_threshold_predicate(X, Y, feature_index, impurity):
    Xs, Ys = zip(*sorted(zip([x[feature_index] for x in X], Y)))
    left_counts = Counter()
    right_counts = Counter(Ys)
    best_score = float('inf')
    best_threshold = None
    i = 0
    while i < len(X) - 1:
        feature_value = Xs[i]
        while i < len(X) and Xs[i] == feature_value:
            # Move element from right to left half
            left_counts.update([Ys[i]])
            right_counts.subtract([Ys[i]])
            i += 1
        if i == len(X):
            # Skip no-op split
            break
        assert(sum(left_counts.values()) == i)
        assert(sum(right_counts.values()) == (len(X) - i))
        score = (impurity(left_counts)*i + impurity(right_counts)*(len(X) - i))
        if score < best_score:
            best_score = score
            best_threshold = (Xs[i - 1] + Xs[i])/2
    return best_threshold

def propose_split_predicates(X, Y, impurity, slow=False):
    for i in range(len(X[0])):
        if is_numerical(X[0][i]):
            if slow:
                ordered_values = sorted({x[i] for x in X})
                # Proposing every possible threshold and then splitting the
                # data to evaluate the split takes O(n^2) time.
                for l, r in zip(ordered_values, ordered_values[1:]):
                    midpoint = (l + r)/2.0
                    yield ThresholdPredicate(i, midpoint)
                continue
            # Evaluating thresholds in a linear scan over the sorted data takes
            # O(n*log(n)) time.
            best_threshold = fast_propose_threshold_predicate(X, Y, i, impurity)
            if best_threshold is not None:
                yield ThresholdPredicate(i, best_threshold)
        elif is_categorical(X[0][i]):
            # Sort values to avoid non-determinism from hash table iteration
            # order.
            unique_values = sorted({x[i] for x in X})
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

def score_split(Y_left, Y_right, impurity):
    # IG = I(Y) - p_left*I(Y_left) - p_right*I(Y_right)
    return impurity(Y_left)*len(Y_left) + impurity(Y_right)*len(Y_right)

class DecisionTree:
    def __init__(self, impurity, max_depth):
        self.root = None
        self.impurity = impurity
        self.max_depth = max_depth
        # Alpha value and a replacement terminal node for each branch.
        self.branch_pruning_info = None

    def is_pure(self, Y):
        raise NotImplementedError

    def create_terminal_node(self, Y):
        raise NotImplementedError

    def construct_decision_tree(self, X, Y, num_examples, depth):
        '''Constructs a decision tree from training data X, Y

        Returns:
            root - root of the tree
            size - number of nodes in the tree
            cost - sum over leaf t of p(t)*impurity(t)
        '''
        node_cost = len(Y)/num_examples*self.impurity(Y)
        terminal_node = self.create_terminal_node(Y)
        if self.is_pure(Y) or depth >= self.max_depth:
            return terminal_node, 1, node_cost

        best_score = float('inf')
        best_split = None
        for split_predicate in propose_split_predicates(X, Y, self.impurity):
            X_left, Y_left, X_right, Y_right = partition(X, Y, split_predicate)
            if not X_left or not X_right:
                continue
            score = score_split(Y_left, Y_right, self.impurity)

            if score < best_score:
                best_score = score
                best_split = (split_predicate, X_left, Y_left, X_right, Y_right)

        if not best_split:
            # TODO: test additional stopping criteria
            # - min samples per leaf
            # - min information gain
            # TODO: best first search for to support max leaf nodes
            return terminal_node, 1, node_cost

        split_predicate, X_left, Y_left, X_right, Y_right = best_split
        left_branch, left_size, left_cost = self.construct_decision_tree(
            X_left, Y_left, num_examples, depth + 1)
        right_branch, right_size, right_cost = self.construct_decision_tree(
            X_right, Y_right, num_examples, depth + 1)
        branch_size = left_size + right_size + 1
        branch_cost = left_cost + right_cost
        branch = InteriorNode(split_predicate, left_branch, right_branch)
        alpha = (node_cost - branch_cost)/(branch_size - 1)
        self.branch_pruning_info[id(branch)] = (alpha, terminal_node)
        return branch, branch_size, branch_cost

    def fit(self, X, Y):
        self.branch_pruning_info = {}
        self.root, _, _ = self.construct_decision_tree(
                X, Y, num_examples=len(Y), depth=0)

    def predict(self, X):
        return [self.root.predict(x) for x in X]

    def score(self, X, Y):
        raise NotImplementedError

    def tree_loss(self, root, X, Y):
        raise NotImplementedError

    def cost_complexity_prune(self, alpha):
        def prune(node):
            if type(node) == TerminalNode:
                return node
            node_alpha, terminal_node = self.branch_pruning_info[id(node)]
            if node_alpha <= alpha:
                return terminal_node
            return InteriorNode(
                node.predicate, prune(node.left), prune(node.right))
        return prune(self.root)

    def prune(self, X_validation, Y_validation):
        best_tree = self.root
        best_loss = float('inf')
        alphas = sorted([alpha for alpha, _ in
            self.branch_pruning_info.values()])
        for alpha in alphas:
            tree = self.cost_complexity_prune(alpha)
            loss = self.tree_loss(tree, X_validation, Y_validation)
            if loss <= best_loss:
                best_loss = loss
                best_tree = tree
        # TODO: return a new DecisionTree object instead?
        self.root = best_tree

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

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, impurity=gini_impurity, max_depth=99999):
        # TODO: accept classification loss functions
        def wrapped_impurity(Y):
            # TODO: see if there is a good way to avoid this
            if type(Y) == Counter:
                return impurity(Y)
            else:
                return impurity(Counter(Y))
        super().__init__(wrapped_impurity, max_depth)

    def is_pure(self, Y):
        for y in Y:
            if y != Y[0]:
                return False
        return True

    def create_terminal_node(self, Y, weights=None):
        distribution = compute_probability_distribution(Y, weights)
        # Predict the most common class
        mode = None
        highest_prob = 0
        for y, p in distribution.items():
            if p > highest_prob:
                mode = y
                highest_prob = p
        return TerminalNode(mode, distribution)

    def predict_prob(self, X):
        return [self.root.predict_prob(x) for x in X]

    def score(self, X, Y):
        num_correct = 0
        for prediction, target in zip(self.predict(X), Y):
            num_correct += (prediction == target)
        return num_correct/len(X)

    def tree_loss(self, root, X, Y):
        num_correct = sum(int(root.predict(x) == y) for x, y in zip(X, Y))
        return -num_correct/len(X)

def get_regression_impurity_fn(loss_fn):
    'Creates a regression impurity function from a regression loss function.'
    def impurity(Y):
        if len(Y) == 0:
            return 0
        prior = loss_fn.prior(Y)
        return loss_fn.loss(Y, [prior]*len(Y))
    return impurity

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, loss_fn=MeanSquaredError(), max_depth=99999,
            purity_tolerance=1e-4):
        impurity = get_regression_impurity_fn(loss_fn)
        super().__init__(impurity, max_depth)
        self.loss = loss_fn
        self.purity_tolerance = purity_tolerance

    def is_pure(self, Y):
        if not Y:
            return True
        return max(Y) - min(Y) < self.purity_tolerance

    def create_terminal_node(self, Y):
        return TerminalNode(self.loss.prior(Y))

    def score(self, X, Y):
        return self.tree_loss(self.root, X, Y)

    def tree_loss(self, root, X, Y):
        return self.loss(Y, [root.predict(x) for x in X])

class DecisionStumpClassifier(DecisionTreeClassifier):
    def __init__(self, impurity=gini_impurity):
        super().__init__(impurity, max_depth=1)

class DecisionStumpRegressor(DecisionTreeRegressor):
    def __init__(self, loss_fn=MeanSquaredError()):
        super().__init__(loss_fn, max_depth=1)
