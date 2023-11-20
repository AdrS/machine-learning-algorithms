import argparse
import math
import pandas as pd

class FieldSummary:

    def __init__(self, series):
        self.description = series.describe()
        self.description['percent_missing'] = \
            float(series.isnull().sum()/len(series))
        self.description['type'] = series.dtype

def is_numerical(series):
    return series.dtype == float or series.dtype == int or series.dtype == bool

def is_categorical(series, max_unique_values=100):
    if series.dtype == float:
        return False
    if series.dtype == int and series.unique().size > max_unique_values:
        return False
    # Bool, string, ...
    return True

def infer_categorical_fields(X, max_categorical_int_unique_values=100):
    for field in X.columns:
        if is_categorical(X[field]):
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
        super().__init__(series)
        self.most_frequent = series.value_counts(dropna=False).sort_values(
            ascending=False)
        self.most_frequent /= len(series)
        self.description['entropy'] = \
          -self.most_frequent.map(lambda x: x*math.log2(x)).sum()

    def report(self, reporter):
        reporter.report_dataframe(self.description[
            ['type', 'percent_missing', 'unique', 'entropy']])
        reporter.report_heading('Most frequent values', 3)
        reporter.report_dataframe(self.most_frequent)

class NumericalFieldSummary(FieldSummary):

    def __init__(self, series):
        super().__init__(series)

    def report(self, reporter):
        reporter.report_dataframe(self.description[
            ['type', 'percent_missing', 'mean', 'std', 'min', '25%', '50%',
            '75%', 'max']])


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--missing_header', action='store_true')
    args = parser.parse_args()
    kwargs = {}
    if args.missing_header:
        kwargs['header'] = None
    dataset = pd.read_csv(args.dataset, **kwargs)
    infer_categorical_fields(dataset)

    DatasetSummary(dataset).report(StdoutReporter())
