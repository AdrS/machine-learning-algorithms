import argparse
import math
import numpy as np
import pandas as pd
from data_transform import pick_smallest_datatypes

class FieldSummary:

    def __init__(self, series, find_most_frequent=False):
        self.description = series.describe()
        self.description['percent_missing'] = \
            float(series.isnull().sum()/len(series))
        self.description['type'] = series.dtype

        self.most_frequent = None
        if find_most_frequent:
            self.most_frequent = series.value_counts(dropna=False).sort_values(
                ascending=False)/len(series)
            self.description['entropy'] = \
              -self.most_frequent.map(lambda x: x*math.log2(x)).sum()

def is_numerical(series):
    return series.dtype == float or series.dtype == int or series.dtype == bool

def set_categorical_fields(X, max_categorical_int_unique_values=20):
    for field in X.columns:
        if X[field].dtype == 'object':
            X[field] = X[field].astype('category')

class SummaryReporter:

    def report_dataframe(self, df, max_rows=None):
        raise NotImplementedError

    def report_heading(self, heading, level=None):
        raise NotImplementedError

class StdoutReporter:

    def report_dataframe(self, df, max_rows=None):
        print(df.to_string())

    def report_heading(self, heading, level=None):
        if level:
            # Number of blank lines between sections is dependent on the
            # heading level.
            for _ in range(max(0, 4 - level)):
                print('')
        print(heading + ':')

class CategoricalFieldSummary(FieldSummary):

    def __init__(self, series):
        super().__init__(series, find_most_frequent=True)

    def report(self, reporter):
        reporter.report_dataframe(self.description[
            ['type', 'percent_missing', 'unique', 'entropy']])
        reporter.report_heading('Most frequent values', 3)
        reporter.report_dataframe(self.most_frequent)

class NumericalFieldSummary(FieldSummary):

    def __init__(self, series, max_unique_values=100):
        # Integer fields with few values might be storing categorical data.
        # Report the most frequent values.
        # Note: this does not detect integer fields with missing values that
        # are encoded as float to use NaN to represent a missing value.
        find_most_frequent = (np.issubdtype(series.dtype, np.integer) and
            series.unique().size < max_unique_values)
        super().__init__(series, find_most_frequent)

    def report(self, reporter):
        fields = ['type', 'percent_missing', 'mean', 'std', 'min', '25%',
            '50%', '75%', 'max']
        if 'entropy' in self.description:
            fields += ['entropy']
        reporter.report_dataframe(self.description[fields])
        if self.most_frequent is not None:
            reporter.report_heading('Most frequent values', 3)
            reporter.report_dataframe(self.most_frequent)


def create_description_dataframe(attributes):
    return pd.concat([pd.Series([value], index=[name])
        for name, value in attributes.items()])

class DatasetSummary:

    def __init__(self, X):
        self.description = create_description_dataframe(
            {'num_fields':X.columns.size, 'num_records':X.shape[0]})
        self.fields = {}
        for field in X.columns:
            is_categorical = type(X[field].dtype) ==  pd.CategoricalDtype
            new_field_sumary = (CategoricalFieldSummary if is_categorical else
                NumericalFieldSummary)
            self.fields[field] = new_field_sumary(X[field])

    def report(self, reporter):
        reporter.report_heading(f'Dataset Summary', 1)
        reporter.report_dataframe(self.description)

        for field, summary in self.fields.items():
            reporter.report_heading(f'Field {field}', 2)
            summary.report(reporter)

def load_dataset(path, missing_header, optimize_memory=False):
    kwargs = {}
    if missing_header:
        kwargs['header'] = None

    if optimize_memory:
        # Optimize the column datatypes using a sample of the dataset
        partial_dataset = next(
            pd.read_csv(args.dataset, chunksize=1000 ,**kwargs))
        pick_smallest_datatypes(partial_dataset)
        kwargs['dtype'] = dict(partial_dataset.dtypes)
        del partial_dataset

    dataset = pd.read_csv(args.dataset, **kwargs)
    set_categorical_fields(dataset)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--missing_header', action='store_true')
    parser.add_argument('--optimize_memory', action='store_true')
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, args.missing_header,
        args.optimize_memory)

    DatasetSummary(dataset).report(StdoutReporter())
