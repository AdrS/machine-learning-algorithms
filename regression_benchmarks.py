import decision_tree

from benchmark import Benchmark, CsvDataset, Field, run_benchmarks, main
from gradient_boost import GradientBoostRegressor
from loss import MeanSquaredError
from sklearn.model_selection import train_test_split

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

# TODO: refactor to use loss module
def squared_error(y, y_pred):
    return (y - y_pred)**2

def absolute_error(y, y_pred):
    return abs(y - y_pred)

class RegressionBenchmark(Benchmark):
    def __init__(self, dataset, loss_fn):
        super().__init__(dataset)
        self.loss_fn = loss_fn

    def show_split_statistics(self, X, Y):
        mean = sum(Y)/len(Y)
        variance = sum(y*y for y in Y)/len(Y) - mean*mean
        print('Target mean', mean)
        print('Target variance', variance)
        print('Num features:', len(X[0]))
        print('Num examples:', len(Y))
        # TODO: print correlation between features

    def evaluate_predictions(self, Y_target, Y_pred):
        return {'loss':self.loss_fn(Y_pred, Y_target)}

    def show_evaluation(self, evaluation_results):
        print('Loss:', evaluation_results['loss'])

class DecisionTreeRegressor:

    def __init__(self, *args, **kwargs):
        self.model = decision_tree.DecisionTreeRegressor(*args, **kwargs)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

class PrunedDecisionTreeRegressor(DecisionTreeRegressor):

    def fit(self, X, Y):
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
            test_size=0.25, random_state=2001)
        self.model.fit(X_train, Y_train)
        self.model.prune(X_val, Y_val)

class GBDT:

    def __init__(self):
        self.model = GradientBoostRegressor(
            create_base_model=lambda *args, **kwargs: PrunedDecisionTreeRegressor(*args, **kwargs),
            loss_fn=MeanSquaredError(),
            num_models=50,
            sample_fraction=None)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)


# TODO: add flag to control loss function
benchmarks = {
    'boston-housing': RegressionBenchmark(
        BostonHousing(), MeanSquaredError()),
    'california-housing': RegressionBenchmark(
        CaliforniaHousing(), MeanSquaredError()),
}
models = {
    'DecisionStump': decision_tree.DecisionStumpRegressor,
    'DecisionTree': DecisionTreeRegressor,
    'PrunedDecisionTree': PrunedDecisionTreeRegressor,
    'GBDT': GBDT,
}

if __name__ == '__main__':
    main(benchmarks, models)
