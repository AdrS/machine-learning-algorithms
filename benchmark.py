import csv
import decision_tree

from sklearn.model_selection import train_test_split

def load_census(path='data/census-income/adult.data'):
    X, Y = [], []
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if not row:
                continue
            x = (
                int(row[0]), # Age
                row[1], # Work class
                row[3], # Education
                int(row[4]), # Education num
                row[5], # Marital status
                row[6], # Occupation
                row[7], # Relationship
                row[8], # Race
                row[9], # Sex
                int(row[10]), # Captial gain
                int(row[11]), # Captial loss
                int(row[12]), # Hours-per-week
                row[13] # Native country
                )
            y = int(row[14].strip() == '>50K')
            X.append(x)
            Y.append(y)
    return X, Y

if __name__ == '__main__':
    X, Y = load_census()
    feature_names = [
        'Age',
        'Work class',
        'Education',
        'Education num',
        'Marital status',
        'Occupation',
        'Relationship',
        'Race',
        'Sex',
        'Captial gain',
        'Captial loss',
        'Hours-per-week',
        'Native country',
    ]

    # TODO: optimize implementation to support larger datasets
    max_dataset_size = 1000
    # TODO: fix the rng seed
    X_train, X_val, Y_train, Y_val = train_test_split(X[:1000], Y[:1000],
        test_size=0.25, random_state=2001)

    model = decision_tree.DecisionTreeClassifier(max_depth=6)
    model.fit(X_train, Y_train)
    print('Model:')
    print(model.export_text(feature_names))
    print('Training Accuracy', model.score(X_train, Y_train))
    print('Validation Accuracy', model.score(X_val, Y_val))
