import argparse
import logging
import os
import pandas as pd
import seaborn as sns
import shutil

from data_exploration import load_dataset, HtmlReporter, create_description_dataframe, NumericalFieldSummary, CategoricalFieldSummary
from pandas.api.types import is_numeric_dtype

def divide_or_nan(numerator, denominator):
    if denominator == 0:
        if numerator == 0:
            return 0
        return float('nan')
    return numerator/denominator

class FieldComparison:
    def __init__(self, train_summary, test_summary):
        train_description = train_summary.description
        test_description = test_summary.description

        # Compare descriptions
        numeric_index = []
        deltas = []
        percent_deltas = []
        for i in train_description.index:
            if not pd.api.types.is_number(train_description[i]):
                continue
            numeric_index.append(i)
            deltas.append(train_description[i] - test_description[i])
            percent_deltas.append(
                divide_or_nan(
                (train_description[i] - test_description[i]),
                (train_description[i] + test_description[i])/2))

        deltas_column = pd.Series(deltas, numeric_index, name='delta')
        percent_deltas_column = pd.Series(percent_deltas, numeric_index,
          name='percent delta')
        self.comparison = pd.concat([
            train_description.rename('train'),
            test_description.rename('test'),
            deltas_column,
            percent_deltas_column
            ], axis=1)

        # Compare value frequencies
        self.most_frequent_train = train_summary.most_frequent
        self.most_frequent_test = test_summary.most_frequent

    def compare_most_frequent(self, reporter, max_unique_values=50):
        if self.most_frequent_train is None or self.most_frequent_test is None:
            return
        # Normalize the series to sum to 1
        train_value_proportions = self.most_frequent_train.rename('train')/self.most_frequent_train.sum()
        test_value_proportions = self.most_frequent_test.rename('test')/self.most_frequent_test.sum()

        # reset_index converts the series to a DataFrame with a column for the
        # series index (in this case the feature values).
        value_proportions = pd.merge(train_value_proportions.reset_index(),
            test_value_proportions.reset_index(), how='outer')
        value_proportions['train'].fillna(0, inplace=True)
        value_proportions['test'].fillna(0, inplace=True)
        value_proportions.sort_values(by=['train', 'test'], ascending=False)
        if value_proportions.shape[0] <= max_unique_values:
            value_proportions.plot(x='index', y=['train', 'test'],
              kind='bar').set(xlabel='Value', ylabel='Proportion')
            reporter.save_figure()
        else:
            sns.lineplot(data=value_proportions[['train', 'test']]).set(
            xlabel='Training value frequency rank',
            ylabel='Proportion')
            reporter.save_figure()

class NumericalFieldComparison(FieldComparison):

    def __init__(self, train_series, test_series):
        super().__init__(
          NumericalFieldSummary(train_series),
          NumericalFieldSummary(test_series))
        self.train_series = train_series
        self.test_series = test_series

    def report(self, reporter):
        fields = ['type', 'percent_missing', 'mean', 'std', 'min', '25%',
            '50%', '75%', 'max']
        if 'entropy' in self.comparison:
            fields += ['entropy']
        reporter.report_dataframe(self.comparison)
        sns.displot(data=[
            self.train_series.rename('train'),
            self.test_series.rename('test')], kind='ecdf').set(
            title=f'{self.train_series.name} CDF',
            xlabel=self.train_series.name)
        reporter.save_figure()
        self.compare_most_frequent(reporter)

class CategoricalFieldComparison(FieldComparison):

    def __init__(self, train_series, test_series):
        super().__init__(
          CategoricalFieldSummary(train_series),
          CategoricalFieldSummary(test_series))

    def report(self, reporter):
        reporter.report_dataframe(self.comparison)
        self.compare_most_frequent(reporter)

class TrainOnlyField:

    def report(self, reporter):
        reporter.print('Only in training dataset')

class TestOnlyField:

    def report(self, reporter):
        reporter.print('Only in test dataset')

class DatasetComparison:
    def __init__(self, X_train, X_test):
        logging.info('Creating dataset comparison')
        self.description = create_description_dataframe(
            {'num_train_records':X_train.shape[0],
            'num_test_records':X_test.shape[0]})
        self.fields = {}
        for field in X_train.columns:
            if field not in X_test.columns:
                self.fields[field] = TrainOnlyField()
                continue
            is_categorical = type(X_train[field].dtype) ==  pd.CategoricalDtype
            new_field_comparison = (
                CategoricalFieldComparison if is_categorical else
                NumericalFieldComparison)
            self.fields[field] = new_field_comparison(X_train[field],
                X_test[field])
        for field in X_test.columns:
            if field not in X_train.columns:
                self.fields[field] = TestOnlyField()
        logging.info('Finished computing comparison')

    def report(self, reporter):
        logging.info('Reporting comparison')
        reporter.report_heading(f'Dataset Comparison', 1)
        reporter.report_dataframe(self.description)

        for field, summary in self.fields.items():
            logging.info(f'Reporting comparison for field {field}')
            reporter.report_heading(f'Field {field}', 2)
            summary.report(reporter)

        reporter.finish()

        logging.info('Finished reporting comparison')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Generates a report comparing the distribution of two datasets')
    parser.add_argument('--train_dataset', required=True,
        help='Path to a CSV file containing the training dataset')
    parser.add_argument('--test_dataset', required=True,
        help='Path to a CSV file containing the training dataset')
    parser.add_argument('--missing_header', action='store_true',
        help='Whether the first line of the CSV file is the header')
    parser.add_argument('--output_directory', required=True,
        help='Directory to save the report to')
    parser.add_argument('--wipe_existing_output', action='store_true',
        help='Whether to delete any existing files in the output directory')
    args = parser.parse_args()
    if args.wipe_existing_output:
        shutil.rmtree(args.output_directory)
    os.makedirs(args.output_directory, exist_ok=True)
    train_dataset = load_dataset(args.train_dataset, args.missing_header)
    test_dataset = load_dataset(args.test_dataset, args.missing_header)

    DatasetComparison(train_dataset, test_dataset).report(
        HtmlReporter(args.output_directory))
