import argparse
import itertools
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

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

def print_evaluation(model, X, Y):
    prediction_proba = model_pipeline.predict_proba(X)
    # TODO: use get_scorer
    score = roc_auc_score(Y, prediction_proba[:, 1])
    print('Score (roc_auc):', score)
    threshold = 0.5
    predictions = prediction_proba[:, 1] >= threshold
    print(f'Confusion Matrix (threshold = {threshold}):')
    print(pd.DataFrame(
            confusion_matrix(Y, predictions, normalize='all'),
            index=[f'Actual: {i}' for i in [False, True]],
            columns=[f'Predicted: {i}' for i in [False, True]]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',
        help='Path to a CSV file containing the dataset',
        default='data/playground-series-s4e1/train.csv')
    parser.add_argument('--target',
        help='Name of the target column',
        default='Exited')
    parser.add_argument('--dropped_features', nargs='+',
        help='List of columns for features to drop before training',
        default=['id', 'CustomerId', 'Surname'])
    parser.add_argument('--numerical_features', nargs='+',
        help='List of columns for numerical features',
        default=['CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'EstimatedSalary'])
    parser.add_argument('--categorical_features', nargs='+',
        help='List of columns for categorical features',
        default=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])
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

    column_transformer = ColumnTransformer([
        ('Categorical', OneHotEncoder(drop='if_binary'),
            list_categorical_features()),
        ('Numerical', StandardScaler(), list_numerical_features())
    ])

    train_dataset = pd.read_csv(args.train_data)
    Y = train_dataset[args.target]
    X = train_dataset.drop(columns=[args.target] + args.dropped_features)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2,
                                        random_state=seed)

    # Cache the preprocessed training data
    X_train_all_features = feature_engineer.fit_transform(X_train)
    X_val_all_features = feature_engineer.fit_transform(X_val)

    # Model family selection
    model_families = {
        'LogisticRegression': LogisticRegression(),
        'MLPClassifier': MLPClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'SVC': SVC(),
    }
    for model_family in args.model_families:
        model = model_families[model_family]
        model_pipeline = Pipeline([
            ('Column transform', column_transformer),
            ('Model', model)
        ])
        print('\n\nEvaluating model family:', model_family)
        # TODO: feature selection
        X_train_features = X_train_all_features
        X_val_features = X_val_all_features

        model_pipeline.fit(X_train_features, Y_train)

        print('Training scores:')
        print_evaluation(model_pipeline, X_train_features, Y_train)

        print('\nValidation scores:')
        print_evaluation(model_pipeline, X_val_features, Y_val)

        print('\nMost important features (permutation importance):')
        print(most_important_features(model_pipeline, X_val_features, Y_val,
            scoring=scoring))

    # Select the best and run hyper parameter searches