import argparse
import itertools
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from functools import reduce
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

def parse_datatype_overrides(overrides):
    # TODO: input validation
    datatype_overrides = []
    for override in overrides:
        column, dtype = override.split('=')
        datatype_overrides.append((column, dtype))
    return datatype_overrides

def load_dataset(path, datatype_overrides):
    dataset = pd.read_csv(path)
    for column, dtype in datatype_overrides:
        dataset[[column]] = dataset[[column]].astype(dtype)
    return dataset

def list_pairs(elements):
    'Returns a list of all unique pairs of elements'
    return itertools.combinations(elements, 2)

class FeatureEngineer:
    def __init__(self, numerical_features=[], log=True, square=True, cube=True,
            categorical_features=[], pairs=True):
        self.numerical_features = numerical_features
        self.log = log
        self.square = square
        self.cube = cube
        self.engineered_numerical_features = []
        for feature in numerical_features:
            if log:
                self.engineered_numerical_features.append(f'log({feature})')
            if square:
                self.engineered_numerical_features.append(f'{feature}^2')
            if cube:
                self.engineered_numerical_features.append(f'{feature}^3')

        self.categorical_features = categorical_features
        self.pairs = pairs
        self.engineered_categorical_features = []
        if pairs:
            for (f1, f2) in list_pairs(categorical_features):
                self.engineered_categorical_features.append(f'({f1}, {f2})')

    def __call__(self, X):
        for feature in self.numerical_features:
            if self.log:
                X[f'log({feature})'] = 0
                mask = X[feature] + 1 > 0
                X[f'log({feature})'] = np.log(X[feature][mask] + 1)
            if self.square:
                X[f'{feature}^2'] = X[feature]**2
            if self.cube:
                X[f'{feature}^3'] = X[feature]**3
        if self.pairs:
            for (f1, f2) in list_pairs(self.categorical_features):
                X[f'({f1}, {f2})'] = [hash(t) for t in zip(X[f1], X[f2])]
        return X

def most_important_features(model, X_val, Y_val, **kwargs):
    r = permutation_importance(model, X_val, Y_val, **kwargs)
    return pd.DataFrame({
        'mean':r.importances_mean,
        'std':r.importances_std},
        index=X_val.columns).sort_values(by=['mean'], ascending=False)

def print_evaluation(X, Y_target, proba):
    # TODO: use get_scorer
    score = roc_auc_score(Y_target, proba)
    print('Score (roc_auc):', score)
    threshold = 0.5
    Y_pred = proba >= threshold
    print(f'Confusion Matrix (threshold = {threshold}):')
    print(pd.DataFrame(
            confusion_matrix(Y_target, Y_pred, normalize='all'),
            index=[f'Actual: {i}' for i in [False, True]],
            columns=[f'Predicted: {i}' for i in [False, True]]))

def subcategory_evaluation(X, Y_target, proba, categorical_features):
    '''
    Compare the model performance for different subcategories to identify
    classes of input the model performs poorly on.
    '''
    for feature in categorical_features:
        print(f'\n\nEvaluation for {feature} values:')
        # TODO: sort values by score
        for value in np.sort(X_train[feature].unique()):
            print('\nValue:', value)
            indicies = X[feature] == value
            X_subset = X[indicies]
            Y_target_subset = Y_target[indicies]
            proba_subset = proba[indicies]
            print_evaluation(X[indicies], Y_target[indicies], proba[indicies])
    # TODO: for numerical features look at correlation between feature value
    # and output score
    # TODO: filter to only show poorly performing subcategories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',
        help='Path to a CSV file containing the dataset',
        default='data/playground-series-s4e1/train.csv')
    parser.add_argument('--train_size', type=int,
        help='Amount of the data to train on during model exploration.' +
            'Use this to speed up the exploration of different models.')
    parser.add_argument('--target',
        help='Name of the target column',
        default='Exited')
    parser.add_argument('--dropped_features', nargs='+',
        help='List of columns for features to drop before training',
        default=['id', 'CustomerId', 'Surname'])
    parser.add_argument('--numerical_features', nargs='*',
        help='List of columns for numerical features',
        default=['CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'EstimatedSalary'])
    parser.add_argument('--categorical_features', nargs='*',
        help='List of columns for categorical features',
        default=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])
    parser.add_argument('--datatype_overrides', nargs='*',
        help='Override the inferred datatypes for the columns',
        default=['HasCrCard=int64', 'IsActiveMember=int64'])
    parser.add_argument('--model_families', nargs='+',
        help='List of model families to try as a first pass',
        default=[
            'LogisticRegression',
            'MLPClassifier',
            'AdaBoostClassifier',
            'RandomForestClassifier',
            'GradientBoostingClassifier',
            'SVC'
        ])
    parser.add_argument('--subcategory_evaluation', action='store_true',
        help='Evaluate model performance for subcategories of the input')
    parser.add_argument('--feature_importance', action='store_true',
        help='Evaluate feature importance using permutation importance')
    args = parser.parse_args()

    seed = 2024
    scoring = 'roc_auc'

    # Feature engineering
    feature_engineer = FunctionTransformer(FeatureEngineer(
        numerical_features=args.numerical_features,
        log=True,
        square=True,
        cube=True,
        categorical_features=args.categorical_features,
        pairs=True))

    def list_numerical_features():
        return args.numerical_features +\
            feature_engineer.func.engineered_numerical_features

    def list_categorical_features():
        return args.categorical_features + \
            feature_engineer.func.engineered_categorical_features

    train_dataset = load_dataset(args.train_data,
        parse_datatype_overrides(args.datatype_overrides))

    Y = train_dataset[args.target]
    X = train_dataset.drop(columns=[args.target] + args.dropped_features)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2,
                                        random_state=seed)

    if args.train_size is not None and args.train_size < len(X_train):
        X_train = X_train[:args.train_size]
        Y_train = Y_train[:args.train_size]

    # Cache the preprocessed training data
    X_train_all_features = feature_engineer.fit_transform(X_train)
    X_val_all_features = feature_engineer.fit_transform(X_val)

    # Model family selection
    model_families = {
        'LogisticRegression': LogisticRegression(random_state=seed),
        'MLPClassifier': MLPClassifier(random_state=seed),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=seed),
        'RandomForestClassifier': RandomForestClassifier(random_state=seed),
        'GradientBoostingClassifier': GradientBoostingClassifier(
            random_state=seed),
        'SVC': SVC(random_state=seed, probability=True),
        'CatBoostClassifier': CatBoostClassifier(
            cat_features=list_categorical_features(),
            random_seed=seed, logging_level="Silent"),
    }
    tree_models = {
        'AdaBoostClassifier',
        'RandomForestClassifier',
        'GradientBoostingClassifier',
        'CatBoostClassifier'
    }
    for model_family in args.model_families:
        model = model_families[model_family]
        steps = []

        # Decision tree models perform better without one-hot encoding and are
        # invariant to scaling of numerical features.
        if model_family == 'CatBoostClassifier':
            # https://catboost.ai/en/docs/concepts/faq#why-float-and-nan-values-are-forbidden-for-cat-features
            # Categorical features must have integer or string datatypes. Use
            # the --datatype_overrides flag to cast the column datatypes as
            # needed.
            pass
        elif model_family in tree_models:
            steps.append(('Column transform',
                ColumnTransformer([
                ('Categorical', OrdinalEncoder(), list_categorical_features()),
                ('Numerical', 'passthrough', list_numerical_features())
            ])))
        else:
            steps.append(('Column transform',
                ColumnTransformer([
                ('Categorical', OneHotEncoder(drop='if_binary'),
                    list_categorical_features()),
                ('Numerical', StandardScaler(), list_numerical_features())
            ])))
        steps.append(('Model', model))
        model_pipeline = Pipeline(steps)

        print('\n\nEvaluating model family:', model_family)
        # TODO: feature selection
        X_train_features = X_train_all_features
        X_val_features = X_val_all_features

        model_pipeline.fit(X_train_features, Y_train)

        proba_train = model_pipeline.predict_proba(X_train_features)[:, 1]
        proba_val = model_pipeline.predict_proba(X_val_features)[:, 1]

        print('Training scores:')
        print('-'*80)
        print_evaluation(X_train_features, Y_train, proba_train)
        if args.subcategory_evaluation:
            subcategory_evaluation(X_val_features, Y_val, proba_val,
                list_categorical_features())

        print('\nValidation scores:')
        print('-'*80)
        print_evaluation(X_val_features, Y_val, proba_val)
        if args.subcategory_evaluation:
            subcategory_evaluation(X_val_features, Y_val, proba_val,
                list_categorical_features())

        if args.feature_importance:
            print('\nMost important features (permutation importance):')
            print(most_important_features(model_pipeline, X_val_features, Y_val,
                scoring=scoring))

    # Select the best and run hyper parameter searches
