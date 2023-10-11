import argparse
import classification_benchmarks
import regression_benchmarks

from bagging import BaggingClassifier, BaggingRegressor
from benchmark import run_benchmarks, select_items

def bagging_benchmarks(benchmarks, models, create_bagging_model, num_models):
    num_models.sort(reverse=True)
    for benchmark_name, benchmark in benchmarks.items():
        print('Benchmark: ', benchmark_name)
        print('#'*80)
        benchmark.initialize()
        benchmark.show_dataset_statistics()
        X_train, Y_train = benchmark.get_training_data()
        X_test = benchmark.get_test_input()

        def show_eval(model):
            print('Training:')
            Y_train_pred = model.predict(X_train)
            benchmark.show_evaluation(
                benchmark.evaluate_predictions(Y_train, Y_train_pred))

            print('Test:')
            Y_test_pred = model.predict(X_test)
            benchmark.show_evaluation(
                benchmark.evaluate_test_predictions(Y_test_pred))

        for model_name, create_model in models.items():
            print('')
            print('Baseline Model:', model_name)
            print('-'*80)
            print('Control (no bagging)')
            control_model = create_model()
            control_model.fit(X_train, Y_train)
            show_eval(control_model)

            bagging_model = create_bagging_model(create_model, num_models[0])
            bagging_model.fit(X_train, Y_train)

            for n in num_models:
                bagging_model.models = bagging_model.models[:n]
                print('')
                print('Bagging', n, 'models')
                show_eval(bagging_model)
                
        print('#'*80)
        print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_benchmarks',
        choices=list(classification_benchmarks.benchmarks.keys()),
        default=[], nargs='*')
    parser.add_argument('--classification_models',
        choices=list(classification_benchmarks.models.keys()),
        default=[], nargs='*')
    parser.add_argument('--regression_benchmarks',
        choices=list(regression_benchmarks.benchmarks.keys()),
        default=[], nargs='*')
    parser.add_argument('--regression_models',
        choices=list(regression_benchmarks.models.keys()),
        default=[], nargs='*')
    parser.add_argument('--num_models', type=int, default=[50], nargs='*')
    args = parser.parse_args()

    bagging_benchmarks(
        select_items(classification_benchmarks.benchmarks,
            args.classification_benchmarks),
        select_items(classification_benchmarks.models,
            args.classification_models),
        BaggingClassifier, args.num_models)

    bagging_benchmarks(
        select_items(regression_benchmarks.benchmarks,
            args.regression_benchmarks),
        select_items(regression_benchmarks.models,
            args.regression_models),
        BaggingRegressor, args.num_models)
