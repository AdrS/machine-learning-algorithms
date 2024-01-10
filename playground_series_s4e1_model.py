import numpy as np
import pandas as pd
from collections import defaultdict
from functools import reduce
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

class SparseToDense:
    def transform(self, X):
        return X.toarray()

def fit_transform(X,
    dropped_features=[],
    numerical_features=[],
    categorical_features=[]):
    # Use sklearn pipeline instead?
    transforms = defaultdict(list)
    for feature in numerical_features:
        scaler = StandardScaler()
        scaler.fit(X[[feature]])
        transforms[feature].append(scaler)

    # TODO: add other transformation like log, sqrt, square
    # TODO: make which transforms to apply a hyper-parameter

    for feature in categorical_features + binary_features:
        encoder = OneHotEncoder(drop='if_binary')
        encoder.fit(X[[feature]])
        transforms[feature].append(encoder)
        transforms[feature].append(SparseToDense())

    def transform(X):
        X = X.drop(columns=dropped_features)
        X_parts = []
        for feature in X.columns:
            X_parts.append(
                reduce(lambda x, t: t.transform(x),
                    transforms[feature], X[[feature]]))
        return np.hstack(X_parts)
    return transform

# TODO: feature engineering transform to combine features

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

    train_dataset = pd.read_csv(train_path)
    Y = train_dataset[target]
    X = train_dataset.drop(columns=[target])

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
    transform = fit_transform(X_train, dropped_features, numerical_features, categorical_features)

    X_train = transform(X_train)
    X_val = transform(X_val)

    # Try different families of models as a first pass
    classifiers = [
        LogisticRegression(),
        MLPClassifier(),
        AdaBoostClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        LinearSVC(),
        SVC(),
    ]
    for classifier in classifiers:
        print('Fitting classifier:', classifier)
        classifier.fit(X_train, Y_train)

        predictions = classifier.predict_proba(X_val)
        score = roc_auc_score(Y_val, predictions[:, 1])
        print('Score:', score)

    # Select the best and run hyper parameter searches
