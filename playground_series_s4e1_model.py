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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.inspection import permutation_importance

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

    preprocess = ColumnTransformer([
        ('Categorical', OneHotEncoder(drop='if_binary'),
            categorical_features + binary_features),
        ('Numerical', StandardScaler(), numerical_features)
    ])

    train_dataset = pd.read_csv(train_path)
    Y = train_dataset[target]
    X = train_dataset.drop(columns=[target] + dropped_features)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2,
                                        random_state=seed)
    # Cache the preprocessed training data
    X_train_post_processed = preprocess.fit_transform(X_train)
    X_val_post_processed = preprocess.transform(X_val)

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
        print(most_important_features(classifier, X_val, Y_val, preprocess))
        print('\n')

    # Select the best and run hyper parameter searches
