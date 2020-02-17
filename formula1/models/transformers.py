import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DTypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtypes):
        self.dtypes = dtypes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(self.dtypes)


class MappingEncoder(TransformerMixin, BaseEstimator):

    def fit(self, X, y):
        categorical_columns = list(X.columns)
        df = pd.concat([X, y], axis=1)

        self.mapping_ = self.get_mapping(categorical_columns, df, X, y)
        self.fallback_ = self.get_fallback(categorical_columns, df, X, y)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ['mapping_', 'fallback_'])

        X = X.copy()
        for column, value in self.mapping_.items():
            X[column] = X[column].map(value).fillna(self.fallback_)

        return X

    def get_mapping(self, categorical_columns, df, X, y):
        raise NotImplemented('subclasses of MappingEncoder should implement get_mapping')


class MeanEncoder(MappingEncoder):
    def get_mapping(self, categorical_columns, df, X, y):
        return {column: df.groupby(column)[y.name].mean() for column in categorical_columns}

    def get_fallback(self, categorical_columns, df, X, y):
        return y.mean()

class MedianEncoder(MappingEncoder):
    def get_mapping(self, categorical_columns, df, X, y):
        return {column: df.groupby(column)[y.name].median() for column in categorical_columns}

    def get_fallback(self, categorical_columns, df, X, y):
        return y.mean()

class DifferenceEncoder(MappingEncoder):
    def get_mapping(self, categorical_columns, df, X, y):
        return {column: df.groupby(column)[y.name].mean() - ((df[y.name].sum() - df.groupby(column)[y.name].sum()) / (len(df) - df.groupby(column).size()))
                for column in categorical_columns}

    def get_fallback(self, categorical_columns, df, X, y):
        return y.mean()

class CorrelationFilter(TransformerMixin, BaseEstimator):

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.columns_to_drop_ = self.get_columns_to_drop(X, y)
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop_)

    @staticmethod
    def _get_abs_corr(X):
        return X.corr().abs() - np.eye(X.shape[1])

    def get_columns_to_drop(self, X, y):

        cols_to_drop = []
        abs_corr = self._get_abs_corr(X)

        while abs_corr.max().max() > self.threshold:
            highest_corr_cols = list(abs_corr.unstack().idxmax())
            col_to_drop = self.choose_from_two(X, y, highest_corr_cols)

            cols_to_drop.append(col_to_drop)
            X = X.drop(columns=col_to_drop)
            abs_corr = self._get_abs_corr(X)

        return cols_to_drop

    def choose_from_two(self, X, y, cols):
        # Of the two columns that share the highest correlation, drop the one most to the back
        return cols[-1]


class CorrFilterLowVariance(CorrelationFilter):
    def __init__(self):
        super(CorrFilterLowVariance, self).__init__()

    def choose_from_two(self, X, y, cols):
        # Of the two columns that share the highest correlation, drop the one with lowest variance
        return X.loc[:, cols].var().idxmin()


class CorrFilterLowTargetCorrelation(CorrelationFilter):
    def __init__(self):
        super(CorrFilterLowTargetCorrelation, self).__init__()

    def choose_from_two(self, X, y, cols):
        # Of the two columns that share the highest correlation, drop the one with lowest correlation with y
        return pd.concat([X.loc[:, cols], y], axis=1).corr()[y.name].idxmin()


class CorrFilterHighTotalCorrelation(CorrelationFilter):
    def __init__(self):
        super(CorrFilterHighTotalCorrelation, self).__init__()

    def choose_from_two(self, X, y, cols):
        # Of the two columns that share the highest correlation,
        # drop the one with highest correlation with other columns
        return X.corr().loc[:, cols].sum(axis=0).idxmax()