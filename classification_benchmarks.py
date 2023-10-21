import argparse
import decision_tree

from benchmark import Benchmark, CsvDataset, Field, run_benchmarks, main
from collections import Counter
from sklearn.model_selection import train_test_split

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

class ClassificationBenchmark(Benchmark):
    def __init__(self, dataset):
        super().__init__(dataset)

    def show_split_statistics(self, X, Y):
        class_frequencies = Counter(Y)
        print('Num classes:', len(class_frequencies))
        print('Num features:', len(X[0]))
        print('Num examples:', len(Y))
        print('Higest frequency: %f%%' % (
            class_frequencies.most_common(1)[0][1]/len(Y)*100))
        if len(class_frequencies) <= 10:
            for label, freq in class_frequencies.items():
                print('%f%% class %r' % (freq/len(Y)*100, label))

    def evaluate_predictions(self, Y_target, Y_pred):
        num_correct = 0
        for target, prediction in zip(Y_target, Y_pred):
            num_correct += (prediction == target)
        return {'accuracy': num_correct/len(Y_target)}

    def show_evaluation(self, evaluation_results):
        print('Accuracy:', evaluation_results['accuracy'])

class DecisionTreeClassifier:

    def __init__(self):
        self.max_depth=999999
        self.model = decision_tree.DecisionTreeClassifier(
            max_depth=self.max_depth)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

class PrunedDecisionTreeClassifier(DecisionTreeClassifier):

    def fit(self, X, Y):
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
            test_size=0.25, random_state=2001)
        self.model.fit(X_train, Y_train)
        self.model.prune(X_val, Y_val)

benchmarks = {
    'census-income': ClassificationBenchmark(CensusIncomeDataset()),
    'dont-get-kicked': ClassificationBenchmark(DontGetKickedDataset()),
    'amazon-employee-access': ClassificationBenchmark(AmazonEmployeeAccessDataset()),
}
models = {
    'DecisionStump': decision_tree.DecisionStumpClassifier,
    'DecisionTree': DecisionTreeClassifier,
    'PrunedDecisionTree': PrunedDecisionTreeClassifier,
}

if __name__ == '__main__':
    main(benchmarks, models)
