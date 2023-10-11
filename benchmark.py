import argparse
import csv

from sklearn.model_selection import train_test_split

class Field:
    '''Field representing a feature or target value of a dataset.'''

    def __init__(self, name, index, parse=lambda x:x):
        self.name = name
        self.index = index
        self.parse = parse

class Dataset:
    def load(self):
        'Loads and returns a list of inputs and a list of output examples.'''
        raise NotImplementedError

class CsvDataset(Dataset):
    '''Dataset backed by a CSV file.'''

    def __init__(self, path, feature_fields, target_field, skip_header=False):
        self.path = path
        self.feature_fields = feature_fields
        self.target_field = target_field
        self.skip_header = skip_header

    def load(self):
        X, Y = [], []
        with open(self.path, 'r') as f:
            reader = csv.reader(f)
            if self.skip_header:
                next(reader)
            for row in reader:
                if not row:
                    continue
                x = []
                for feature in self.feature_fields:
                    x.append(feature.parse(row[feature.index]))
                y = self.target_field.parse(row[self.target_field.index])
                X.append(tuple(x))
                Y.append(y)
        return X, Y

    def feature_names(self):
        return [field.name for field in self.feature_fields]

class Benchmark:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def initialize(self, test_size=0.25, random_state=2001):
        'Loads benchmark dataset and partitions into training and set splits.'
        X, Y = self.dataset.load()
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(X, Y, test_size=test_size,
                random_state=random_state)

        # TODO: optimize implementation to support larger datasets
        max_dataset_size = 1000
        self.X_train = self.X_train[:max_dataset_size]
        self.Y_train = self.Y_train[:max_dataset_size]

    def show_split_statistics(self, X, Y):
        raise NotImplementedError

    def show_dataset_statistics(self):
        print('Training dataset')
        self.show_split_statistics(self.X_train, self.Y_train)
        print('Test dataset')
        self.show_split_statistics(self.X_test, self.Y_test)

    def get_training_data(self):
        return self.X_train, self.Y_train

    def get_test_input(self):
        return self.X_test

    def evaluate_predictions(self, Y_target, Y_pred):
        raise NotImplementedError

    def evaluate_test_predictions(self, Y_pred):
        return self.evaluate_predictions(self.Y_test, Y_pred)

    def show_evaluation(self):
        raise NotImplementedError

def run_benchmarks(benchmarks, models):
    for benchmark_name, benchmark in benchmarks.items():
        print('Benchmark: ', benchmark_name)
        print('#'*80)
        benchmark.initialize()
        benchmark.show_dataset_statistics()
        X_train, Y_train = benchmark.get_training_data()
        X_test = benchmark.get_test_input()

        for model_name, create_model in models.items():
            print('')
            print('Model:', model_name)
            print('-'*80)
            model = create_model()
            model.fit(X_train, Y_train)

            print('Training:')
            Y_train_pred = model.predict(X_train)
            benchmark.show_evaluation(
                benchmark.evaluate_predictions(Y_train, Y_train_pred))

            print('Test:')
            Y_test_pred = model.predict(X_test)
            benchmark.show_evaluation(
                benchmark.evaluate_test_predictions(Y_test_pred))
        print('#'*80)
        print('')

def select_items(dictionary, keys):
    return {key:value for key, value in dictionary.items() if key in keys}

def main(benchmarks, models):
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', choices=list(benchmarks.keys()),
        default=[], nargs='*')
    parser.add_argument('--models', choices=list(models.keys()),
        default=[], nargs='*')
    args = parser.parse_args()
    run_benchmarks(select_items(benchmarks, args.benchmarks),
        select_items(models, args.models))
