import argparse
import csv
import decision_tree

from collections import Counter
from sklearn.model_selection import train_test_split

class Field:
    def __init__(self, name, index, parse=lambda x:x):
        self.name = name
        self.index = index
        self.parse = parse

class CsvDataset:
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

class CensusIncomeDataset(CsvDataset):
    def __init__(self, path='data/census-income/adult.data'):
        super().__init__(path, feature_fields = [
            Field('Age', 0, int),
            Field('Work class', 1),
            Field('Education', 3),
            Field('Education num', 4, int),
            Field('Marital status', 5),
            Field('Occupation', 6),
            Field('Relationship', 7),
            Field('Race', 8),
            Field('Sex', 9),
            Field('Captial gain', 10, int),
            Field('Captial loss', 11, int),
            Field('Hours-per-week', 12, int),
            Field('Native country', 13)
        ],
        target_field=Field('>50k', 14, lambda x: int(x.strip() == '>50K')),
        skip_header=False)

class DontGetKickedDataset(CsvDataset):
    def __init__(self, path='data/DontGetKicked/training.csv'):
        super().__init__(path, feature_fields = [
            #Field('RefID', 0),
            #Field('IsBadBuy', 1),
            #Field('PurchDate', 2),
            Field('Auction', 3),
            Field('VehYear', 4, int),
            Field('VehicleAge', 5, int),
            Field('Make', 6),
            Field('Model', 7),
            Field('Trim', 8),
            Field('SubModel', 9),
            Field('Color', 10),
            Field('Transmission', 11),
            Field('WheelTypeID', 12),
            Field('WheelType', 13),
            Field('VehOdo', 14, int),
            Field('Nationality', 15),
            Field('Size', 16),
            Field('TopThreeAmericanName', 17),
            # Null values :(
            #Field('MMRAcquisitionAuctionAveragePrice', 18, int),
            #Field('MMRAcquisitionAuctionCleanPrice', 19, int),
            #Field('MMRAcquisitionRetailAveragePrice', 20, int),
            #Field('MMRAcquisitonRetailCleanPrice', 21, int),
            #Field('MMRCurrentAuctionAveragePrice', 22, int),
            #Field('MMRCurrentAuctionCleanPrice', 23, int),
            #Field('MMRCurrentRetailAveragePrice', 24, int),
            #Field('MMRCurrentRetailCleanPrice', 25, int),
            Field('PRIMEUNIT', 26),
            Field('AUCGUART', 27),
            #Field('BYRNO', 28),
            Field('VNZIP', 29),
            Field('VNST', 30),
            Field('VehBCost', 31, float),
            Field('IsOnlineSale', 32),
            Field('WarrantyCost', 33, int),
        ],
        target_field=Field('IsBadBuy', 1, int),
        skip_header=True)

class AmazonEmployeeAccessDataset(CsvDataset):
    def __init__(self, path='data/amazon-employee-access-challenge-train.csv'):
        super().__init__(path, feature_fields = [
            Field('RESOURCE', 1),
            Field('MGR_ID', 2),
            Field('ROLE_ROLLUP_1', 3),
            Field('ROLE_ROLLUP_2', 4),
            Field('ROLE_DEPTNAME', 5),
            Field('ROLE_TITLE', 6),
            Field('ROLE_FAMILY_DESC', 7),
            Field('ROLE_FAMILY', 8),
            Field('ROLE_CODE', 9),
        ],
        target_field=Field('ACTION', 0, int),
        skip_header=True)

class BostonHousing(CsvDataset):
    def __init__(self, path='data/boston-housing/train.csv'):
        super().__init__(path, feature_fields = [
            #Field('ID', 0, float),
            Field('crim', 1, float),
            Field('zn', 2, float),
            Field('indus', 3, float),
            Field('chas', 4, float),
            Field('nox', 5, float),
            Field('rm', 6, float),
            Field('age', 7, float),
            Field('dis', 8, float),
            Field('rad', 9, float),
            Field('tax', 10, float),
            Field('ptratio', 11, float),
            Field('black', 12, float),
            Field('lstat', 13, float),
        ],
        target_field=Field('medv', 14, float),
        skip_header=True)

class CaliforniaHousing(CsvDataset):
    def __init__(self, path='data/CaliforniaHousing/cal_housing.data'):
        super().__init__(path, feature_fields = [
            Field('longitude', 0, float),
            Field('latitude', 1, float),
            Field('housingMedianAge', 2, float),
            Field('totalRooms', 3, float),
            Field('totalBedrooms', 4, float),
            Field('population', 5, float),
            Field('households', 6, float),
            Field('medianIncome', 7, float),
        ],
        target_field=Field('medianHouseValue', 8, float),
        skip_header=False)

class Benchmark:
    def __init__(self, dataset):
        self.dataset = dataset

    def load(self):
        return self.dataset.load()

    def print_dataset_stats(self, X, Y):
        raise NotImplementedError

class ClassificationBenchmark(Benchmark):
    def __init__(self, dataset):
        super().__init__(dataset)

    def print_dataset_stats(self, X, Y):
        class_frequencies = Counter(Y)
        print('Num classes:', len(class_frequencies))
        print('Num features:', len(X[0]))
        print('Num examples:', len(Y))
        print('Higest frequency: %f%%' % (
            class_frequencies.most_common(1)[0][1]/len(Y)*100))
        if len(class_frequencies) <= 10:
            for label, freq in class_frequencies.items():
                print('%f%% class %r' % (freq/len(Y)*100, label))

class RegressionBenchmark(Benchmark):
    def __init__(self, dataset):
        super().__init__(dataset)

    def print_dataset_stats(self, X, Y):
        mean = sum(Y)/len(Y)
        variance = sum(y*y for y in Y)/len(Y) - mean*mean
        print('Target mean', mean)
        print('Target variance', variance)
        print('Num features:', len(X[0]))
        print('Num examples:', len(Y))
        # TODO: print correlation between features

class Model:
    def fit(self, X_train, Y_train):
        raise NotImplementedError

    def eval(self, X_train, Y_train, X_val, Y_val, dataset):
        raise NotImplementedError

class DecisionTreeClassifier(Model):
    def __init__(self):
        self.max_depth=999999
        self.model = decision_tree.DecisionTreeClassifier(
            max_depth=self.max_depth)

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def eval(self, X_train, Y_train, X_val, Y_val, dataset):
        if self.max_depth < 6:
            print('Model:')
            print(self.model.export_text(dataset.feature_names()))
        print('Training Accuracy', self.model.score(X_train, Y_train))
        print('Validation Accuracy', self.model.score(X_val, Y_val))

class DecisionTreeRegressor(Model):
    def __init__(self):
        self.max_depth=999999
        self.model = decision_tree.DecisionTreeRegressor(
            max_depth=self.max_depth)

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def eval(self, X_train, Y_train, X_val, Y_val, dataset):
        if self.max_depth < 6:
            print('Model:')
            print(self.model.export_text(dataset.feature_names()))
        print('Training loss', self.model.score(X_train, Y_train))
        print('Validation loss', self.model.score(X_val, Y_val))


if __name__ == '__main__':
    benchmarks = {
        'census-income': ClassificationBenchmark(CensusIncomeDataset()),
        'dont-get-kicked': ClassificationBenchmark(DontGetKickedDataset()),
        'amazon-employee-access': ClassificationBenchmark(AmazonEmployeeAccessDataset()),
        'boston-housing': RegressionBenchmark(BostonHousing()),
        'california-housing': RegressionBenchmark(CaliforniaHousing()),
    }
    models = {
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'DecisionTreeRegressor': DecisionTreeRegressor
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', choices=list(benchmarks.keys()),
        default=[], nargs='*')
    parser.add_argument('--models', choices=list(models.keys()),
        default=[], nargs='*')
    args = parser.parse_args()
    for benchmark_name in args.benchmarks:
        print('Benchmark:', benchmark_name)
        benchmark = benchmarks[benchmark_name]
        X, Y = benchmark.load()
        benchmark.print_dataset_stats(X, Y)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
            test_size=0.25, random_state=2001)

        # TODO: optimize implementation to support larger datasets
        max_dataset_size = 1000
        X_train, Y_train = X_train[:max_dataset_size], Y_train[:max_dataset_size]
        for model_name in args.models:
            print('\nModel:', model_name)
            model = models[model_name]()
            model.fit(X_train, Y_train)
            model.eval(X_train, Y_train, X_val, Y_val, benchmark.dataset)
        print('\n')
