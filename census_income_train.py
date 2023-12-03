import hyperopt
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool, metrics, cv
from data_exploration import load_dataset

# Adapted from
# https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb
# (APACHE license)

if __name__ == '__main__':
    seed = 2023
    train_dataset = load_dataset('data/census-income/train.csv')
    Y = train_dataset['income'] == ' >50K'
    X = train_dataset.drop(columns=['income'])
    cat_features=[i for i in X.columns if type(X[i].dtype) == pd.CategoricalDtype]

    def create_model(params):
        return CatBoostClassifier(
            l2_leaf_reg=int(params['l2_leaf_reg']),
            learning_rate=params['learning_rate'],
            iterations=100,
            eval_metric=metrics.Accuracy(),
            random_seed=42,
            verbose=False,
            loss_function=metrics.Logloss(),
        )

    def hyperopt_objective(params):
        model = create_model(params)
        cv_data = cv(
            Pool(X, Y, cat_features=cat_features),
            model.get_params(),
            logging_level='Silent',
        )
        best_accuracy = np.max(cv_data['test-Accuracy-mean'])
        return 1 - best_accuracy
    params_space = {
        'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
    }

    trials = hyperopt.Trials()

    best = hyperopt.fmin(
        hyperopt_objective,
        space=params_space,
        algo=hyperopt.tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(seed)
    )
    #best = {'l2_leaf_reg': 1.0, 'learning_rate': 0.27067899458354466}

    print('Best hyperparameter values', best)
    model = create_model(best)
    cv_data = cv(Pool(X, Y, cat_features=cat_features), model.get_params())
    print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))
    model.fit(X, Y, cat_features=cat_features)
    model.save_model('output/models/census-income.dump')
