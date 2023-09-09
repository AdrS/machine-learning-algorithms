import csv
import decision_tree

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


if __name__ == '__main__':
    #dataset = CensusIncomeDataset()
    dataset = DontGetKickedDataset()
    X, Y = dataset.load()

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
        test_size=0.25, random_state=2001)
    # TODO: optimize implementation to support larger datasets
    max_dataset_size = 100
    X_train, Y_train = X_train[:max_dataset_size], Y_train[:max_dataset_size]

    max_depth=999999
    model = decision_tree.DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, Y_train)
    if max_depth < 6:
        print('Model:')
        print(model.export_text(dataset.feature_names()))
    print('Training Accuracy', model.score(X_train, Y_train))
    print('Validation Accuracy', model.score(X_val, Y_val))
