import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import reduce
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

def most_important_features(model, X_val, Y_val, preprocess):
    pipeline = Pipeline([('preprocess', preprocess), ('model', model)])
    r = permutation_importance(pipeline, X_val, Y_val, scoring='roc_auc')
    return pd.DataFrame({
        'mean':r.importances_mean,
        'std':r.importances_std},
        index=X_val.columns).sort_values(by=['mean'], ascending=False)

if __name__ == '__main__':
    # TODO: refactor into a library to use for other datasets
    seed = 2024
    train_path = 'data/playground-series-s4e1/train.csv'
    target = 'Exited'
    dropped_features = ['id', 'CustomerId', 'Surname']
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'EstimatedSalary']
    binary_features = ['HasCrCard', 'IsActiveMember'] 
    categorical_features = ['Geography', 'Gender']

    feature_engineer = FunctionTransformer(FeatureEngineer(
        numerical_features=numerical_features,
        log=True,
        square=True,
        cube=True,
        categorical_features=categorical_features + binary_features,
        pairs=True))

    def list_numerical_features():
        return numerical_features +\
            feature_engineer.func.engineered_numerical_features

    def list_categorical_features():
        return categorical_features + binary_features + \
            feature_engineer.func.engineered_categorical_features

    column_transformer = ColumnTransformer([
        ('Categorical', OneHotEncoder(drop='if_binary'),
            list_categorical_features()),
        ('Numerical', StandardScaler(), list_numerical_features())
    ])
    preprocess = Pipeline([
        ('Feature engineer', feature_engineer),
        ('Column', column_transformer)
    ])

    train_dataset = pd.read_csv(train_path)
    Y = train_dataset[target]
    X = train_dataset.drop(columns=[target] + dropped_features)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2,
                                        random_state=seed)
    # Cache the preprocessed training data
    X_train_post_processed = preprocess.fit_transform(X_train)
    X_val_post_processed = preprocess.transform(X_val)
    X_val_feature_engineered = feature_engineer.transform(X_val)

    # Try different families of models as a first pass
    classifiers = [
        LogisticRegression(),
        MLPClassifier(),
        AdaBoostClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC(),
    ]
    for classifier in classifiers:
        print('Fitting classifier:', classifier)
        classifier.fit(X_train_post_processed, Y_train)

        predictions = classifier.predict_proba(X_val_post_processed)
        score = roc_auc_score(Y_val, predictions[:, 1])
        print('Score (roc_auc):', score)
        print('Most important features (permutation importance):')
        print(most_important_features(classifier, X_val_feature_engineered, Y_val, column_transformer))
        print('\n')

    # Select the best and run hyper parameter searches
