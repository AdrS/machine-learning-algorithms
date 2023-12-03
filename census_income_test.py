from catboost import CatBoostClassifier
from data_exploration import load_dataset
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    test_dataset = load_dataset('data/census-income/test.csv')
    Y = (test_dataset['income'] == ' >50K.').astype('string')
    X = test_dataset.drop(columns=['income'])
    model = CatBoostClassifier()
    model.load_model('output/models/census-income.dump')
    predictions = model.predict(X)
    print('Accuracy:', accuracy_score(Y, predictions))
    #model.predict_proba(X)
    # output prediction probabilities
    # analysis notebook, plot precision recall curve, aoc, confusion matrix, ..
