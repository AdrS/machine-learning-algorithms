import argparse
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutil
import sys
from data_transform import pick_smallest_datatypes
from xml.etree import ElementTree as ET

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
        self.series = series

def set_categorical_fields(X, max_categorical_int_unique_values=20):
    for field in X.columns:
        if X[field].dtype == 'object':
            X[field] = X[field].astype('category')

class SummaryReporter:

    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.num_figures = 0

    def report_dataframe(self, df, max_rows=None):
        raise NotImplementedError

    def report_heading(self, heading, level=None):
        raise NotImplementedError

    def save_figure(self, name=None):
        if not name:
            name = 'fig%d.png' % (self.num_figures,)
            self.num_figures += 1
        plt.savefig(os.path.join(self.output_directory, name))
        plt.close()
        return name

    def print(self, *args):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError

class PlainTextReporter(SummaryReporter):

    def __init__(self, output_directory):
        super().__init__(output_directory)
        self.fd = open(os.path.join(output_directory, 'report.txt'), 'w')

    def report_dataframe(self, df, max_rows=None):
        self.print(df.to_string())

    def report_heading(self, heading, level=None):
        if level:
            # Number of blank lines between sections is dependent on the
            # heading level.
            for _ in range(max(0, 4 - level)):
                self.print('')
        self.print(heading + ':')

    def save_figure(self, name=None):
        path = super().save_figure(name)
        self.print('Figure:', path)

    def print(self, *args):
        print(*args, file=self.fd)

    def finish(self):
        self.fd.close()

class HtmlReporter(SummaryReporter):

    def __init__(self, output_directory):
        super().__init__(output_directory)
        self.body = ET.Element('body')

    def report_dataframe(self, df, max_rows=None):
        pre = ET.Element('pre')
        pre.text = df.to_string()
        self.body.append(pre)

    def report_heading(self, heading, level=None):
        h = ET.Element('h%d' % (level,))
        h.text = heading
        self.body.append(h)

    def save_figure(self, name=None):
        path = super().save_figure(name)
        img = ET.Element('img', src=path)
        self.body.append(img)

    def print(self, *args):
        div = ET.Element('div')
        output = io.StringIO()
        print(*args, file=output)
        div.text = output.getvalue()
        output.close()
        self.body.append(div)

    def finish(self):
        html = ET.Element('html')
        html.append(self.body)
        path = os.path.join(self.output_directory, 'report.html')
        with open(path, 'wb') as f:
            ET.ElementTree(html).write(f, encoding='utf-8', method='html')

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

        sns.displot(self.series)
        reporter.save_figure()

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

        reporter.report_heading('Less useful fields', 2)
        reporter.print('Mostly missing fields:', self.mostly_missing_fields())
        reporter.print('Low entropy fields:', self.low_entropy_fields())
        reporter.finish()

    def mostly_missing_fields(self, threshold=0.9):
        return [field for field, summary in self.fields.items()
            if summary.description['percent_missing'] > threshold]

    def low_entropy_fields(self, threshold=0.07638839439271368):
        return [field for field, summary in self.fields.items()
            if 'entropy' in summary.description and
            summary.description['entropy'] < threshold]

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
    parser = argparse.ArgumentParser(
        description='Generates a report describing the features of the dataset'
    )
    parser.add_argument('--dataset', required=True,
        help='Path to a CSV file containing the dataset')
    parser.add_argument('--missing_header', action='store_true',
        help='Whether the first line of the CSV file is the header')
    parser.add_argument('--optimize_memory', action='store_true')

    parser.add_argument('--output_directory', required=True,
        help='Directory to save the report to')
    parser.add_argument('--wipe_existing_output', action='store_true',
        help='Whether to delete any existing files in the output directory')
    args = parser.parse_args()
    if args.wipe_existing_output:
        shutil.rmtree(args.output_directory)
    os.makedirs(args.output_directory, exist_ok=True)
    dataset = load_dataset(args.dataset, args.missing_header,
        args.optimize_memory)
    DatasetSummary(dataset).report(HtmlReporter(args.output_directory))
