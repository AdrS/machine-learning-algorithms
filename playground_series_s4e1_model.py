import argparse
import itertools
import json
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

def parse_datatype_overrides(overrides):
    # TODO: input validation
    datatype_overrides = []
    for override in overrides:
        column, dtype = override.split('=')
        datatype_overrides.append((column, dtype))
    return datatype_overrides

def load_dataset(path, datatype_overrides):
    dataset = pd.read_csv(path, index_col=False)
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

def create_model_pipeline(params):
    model_family = params['model_family']
    seed = params['seed']
    categorical_features = params['categorical_features']
    numerical_features = params['numerical_features']

    steps = []

    # Preprocessing
    ###########################################################################
    # Feature engineering
    feature_engineer = FunctionTransformer(FeatureEngineer(
        numerical_features=args.numerical_features,
        log=True,
        square=True,
        cube=True,
        categorical_features=args.categorical_features,
        # Model evaluation fails when there is a new combination of categorical
        # feature values at test time.
        pairs=False))
    steps.append(('Feature engineering', feature_engineer))

    numerical_features = args.numerical_features + \
            feature_engineer.func.engineered_numerical_features
    categorical_features = args.categorical_features + \
            feature_engineer.func.engineered_categorical_features

    # Feature transformations
    tree_models = {
        'AdaBoostClassifier',
        'RandomForestClassifier',
        'GradientBoostingClassifier',
        'CatBoostClassifier'
    }
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
            ('Categorical', OrdinalEncoder(), categorical_features),
            ('Numerical', 'passthrough', numerical_features)
        ])))
    else:
        steps.append(('Column transform',
            ColumnTransformer([
            ('Categorical', OneHotEncoder(drop='if_binary'),
                categorical_features),
            ('Numerical', StandardScaler(), numerical_features)
        ])))

    # Model
    ###########################################################################
    if model_family == 'LogisticRegression':
        model = LogisticRegression(random_state=seed)
    elif model_family == 'MLPClassifier':
        model = MLPClassifier(random_state=seed)
    elif model_family == 'AdaBoostClassifier':
        model = AdaBoostClassifier(random_state=seed)
    elif model_family == 'RandomForestClassifier':
        model = RandomForestClassifier(random_state=seed)
    elif model_family == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(random_state=seed)
    elif model_family == 'SVC':
        model = SVC(random_state=seed, probability=True)
    elif model_family == 'CatBoostClassifier':
        model = CatBoostClassifier(
            cat_features=categorical_features,
            random_seed=seed, logging_level="Silent")
    steps.append(('Model', model))

    return Pipeline(steps)

# From: https://stackoverflow.com/questions/26646362
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def pp(obj):
    return json.dumps(obj, sort_keys=True, indent=2, cls=NumpyEncoder)

def log_params(params, enable_mlflow=True, enable_stdout=True):
    if enable_stdout:
        print('Model:', pp(params))
    if enable_mlflow:
        mlflow.log_params(params)

def log_data(X, Y, label, enable_mlflow=True):
    if not enable_mlflow:
        return
    mlflow.log_input(
        mlflow.data.from_pandas(pd.concat([X, Y], axis=1)),
        context=label)

class MlFlowModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

def log_model(model, enable_mlflow=True):
    if not enable_mlflow:
        return
    mlflow.sklearn.log_model(MlFlowModelWrapper(model), 'model')
    #mlflow.catboost.log_model(model, 'model')

def evaluate_performance(Y_target, proba):
    threshold = 0.5
    Y_pred = proba >= threshold
    cm = confusion_matrix(Y_target, Y_pred, normalize='all')
    return {
        'roc_auc':roc_auc_score(Y_target, proba),
        'cm':cm
    }

def log_performance(metrics,
        enable_mlflow=True, metric_suffix='',
        enable_stdout=True):
    roc_auc = metrics['roc_auc']
    if enable_mlflow:
        mlflow.log_metric(f'{metric_suffix}_roc_auc', roc_auc)
        mlflow.log_text(json.dumps(metrics['cm'], cls=NumpyEncoder),
            artifact_file=f'{metric_suffix}_confusion_matrix.json')
    if enable_stdout:
        print(metric_suffix, 'roc auc:', roc_auc)
        print('Confusion matrix:', metrics['cm'])

def evaluate_subcategory_performance(X, Y_target, proba, categorical_features):
    '''
    Compare the model performance for different subcategories to identify
    classes of input the model performs poorly on.
    '''
    metrics = []
    for feature in categorical_features:
        metrics_by_value = []
        for value in np.sort(X[feature].unique()):
            indicies = X[feature] == value
            Y_target_subset = Y_target[indicies]
            proba_subset = proba[indicies]
            value_metrics = evaluate_performance(Y_target_subset, proba_subset)
            value_metrics['value'] = value
            metrics_by_value.append(value_metrics)
        metrics_by_value.sort(key=lambda x: x['roc_auc'])
        feature_metrics = {
            'feature':feature,
            'values':metrics_by_value
        }
        metrics.append(feature_metrics)
    # TODO: for numerical features look at correlation between feature value
    # and output score
    return metrics


def log_subcategory_performance(metrics,
        enable_mlflow=True, metric_suffix='',
        enable_stdout=True):
    if enable_mlflow:
        mlflow.log_text(json.dumps(metrics, cls=NumpyEncoder),
            'subcategory_performance.json')
    if enable_stdout:
        # TODO: filter to only show poorly performing subcategories
        print('Subcategory performance:', pp(metrics))

def evaluate_feature_importance(model, X_val, Y_val, **kwargs):
    r = permutation_importance(model, X_val, Y_val, **kwargs)
    return pd.DataFrame({
        'mean':r.importances_mean,
        'std':r.importances_std},
        index=X_val.columns).sort_values(by=['mean'], ascending=False)

def log_feature_importance(feature_importance, enable_mlflow=True,
        enable_stdout=True):
    if enable_mlflow:
        mlflow.log_table(
            feature_importance.reset_index().rename(columns={'index':'feature'}),
            artifact_file='feature_importance.json')
    if enable_stdout:
        print('Feature importance:\n', feature_importance)

def do_training_runs(args):
    seed = 2024
    scoring = 'roc_auc'

    train_dataset = load_dataset(args.train_data,
        parse_datatype_overrides(args.datatype_overrides))

    Y = train_dataset[args.target]
    X = train_dataset[args.numerical_features + args.categorical_features]

    use_validation_set = args.val_fraction > 0.0
    if use_validation_set:
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
            test_size=args.val_fraction, random_state=seed)
    else:
        X_train, Y_train = shuffle(X, Y, random_state=seed)

    if args.train_size is not None and args.train_size < len(X_train):
        X_train = X_train[:args.train_size]
        Y_train = Y_train[:args.train_size]

    # TODO: feature selection

    # Model family selection
    for model_family in args.model_families:
        with mlflow.start_run():
            params = {
                'model_family':model_family,
                'seed':seed,
                'categorical_features': args.categorical_features,
                'numerical_features': args.numerical_features,
                'train_size':args.train_size,
                'val_fraction':args.val_fraction,
            }
            log_params(params)
            model_pipeline = create_model_pipeline(params)

            log_data(X_train, Y_train, 'train')
            if use_validation_set:
                log_data(X_val, Y_val, 'val')

            model_pipeline.fit(X_train, Y_train)
            log_model(model_pipeline)

            proba_train = model_pipeline.predict_proba(X_train)[:, 1]
            log_performance(evaluate_performance(Y_train, proba_train),
                metric_suffix='train')
            if use_validation_set:
                proba_val = model_pipeline.predict_proba(X_val)[:, 1]
                log_performance(evaluate_performance(Y_val, proba_val),
                    metric_suffix='val')

            if args.subcategory_evaluation:
                feature_engineer = model_pipeline.named_steps[
                    'Feature engineering']
                log_subcategory_performance(
                    evaluate_subcategory_performance(
                        feature_engineer.transform(X_val),
                        Y_val, proba_val, (args.categorical_features +
                        feature_engineer.func.engineered_categorical_features)))

            if args.feature_importance:
                # Engineered features
                log_feature_importance(evaluate_feature_importance(
                    Pipeline(model_pipeline.steps[1:]),
                    model_pipeline.named_steps[
                        'Feature engineering'].transform(X_val),
                    Y_val, scoring=scoring))

    # Select the best and run hyper parameter searches

def do_inference(args):
    dataset = load_dataset(args.inference_input,
        parse_datatype_overrides(args.datatype_overrides))
    X = dataset[args.numerical_features + args.categorical_features]
    model = mlflow.pyfunc.load_model(args.inference_model)
    proba = model.predict(X)
    submission = pd.DataFrame({
        'id':dataset['id'],
        'Exited':proba
    })
    submission.to_csv(args.inference_output, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',
        help='Path to a CSV file containing the dataset',
        default='data/playground-series-s4e1/train.csv')
    parser.add_argument('--train_size', type=int,
        help='Amount of the data to train on during model exploration.' +
            'Use this to speed up the exploration of different models.')
    parser.add_argument('--val_fraction', type=float,
        help='Fraction of training data used for the validation set.' +
            'Set to 0 to use all data for training and none for validation.',
            default=0.2)

    parser.add_argument('--inference_model',
        help='Run id of the model to use for inference')
    parser.add_argument('--inference_input',
        help='Path to a CSV file containing the input to run inference on')
    parser.add_argument('--inference_output',
        help='Path to a CSV output with the final predictions')

    parser.add_argument('--mlflow_tracking_uri',
        help='Address of the experiment tracking server.',
        default='http://127.0.0.1:9999')
    parser.add_argument('--mlflow_experiment_name',
        default='Playground series s4e1')

    parser.add_argument('--target',
        help='Name of the target column',
        default='Exited')
    parser.add_argument('--numerical_features', nargs='*',
        help='List of columns for numerical features',
        default=['CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'EstimatedSalary'])
    parser.add_argument('--categorical_features', nargs='*',
        help='List of columns for categorical features',
        default=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])
    parser.add_argument('--datatype_overrides', nargs='*',
        help='Override the inferred datatypes for the columns',
        default=['HasCrCard=int8', 'IsActiveMember=int8'])
    parser.add_argument('--model_families', nargs='+',
        help='List of model families to try as a first pass',
        default=[
            'LogisticRegression',
            'MLPClassifier',
            'AdaBoostClassifier',
            'RandomForestClassifier',
            'GradientBoostingClassifier',
            'SVC',
            'CatBoostClassifier',
        ])
    parser.add_argument('--subcategory_evaluation', action='store_true',
        help='Evaluate model performance for subcategories of the input')
    parser.add_argument('--feature_importance', action='store_true',
        help='Evaluate feature importance using permutation importance')
    args = parser.parse_args()


    mlflow.set_tracking_uri(uri=args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    if args.inference_model is not None:
        do_inference(args)
    else:
        do_training_runs(args)
