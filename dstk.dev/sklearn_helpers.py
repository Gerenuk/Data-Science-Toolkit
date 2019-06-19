"""
Helper functions specifically for anomaly detection which general and *not* specifically configured for the dataset from the project
"""

from pyod.models.knn import KNN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import pandas as pd
import numpy as np
from collections import namedtuple


ModelSpec2feats = namedtuple("ModelSpec2feats", "model col_name model_on_feats feats")
# Contains:
# model: sklearn model
# col_name: new name of generated prediction column
# model_on_feats: subpart of model pipeline which only does the scaling and prediction; required to faster draw contour plots, which are based on generated raw values
# feats: names of features used in prediction



class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Subselects columns for the Sklearn Pipeline
    needs Pandas DataFrame
    """

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.feature_names].values
    
    
class DefaultNA(BaseEstimator, ClassifierMixin):
    """
    Classifier wrapper which will handle NaN values in the features and predict a `na_default` value for missing values
    """
    def __init__(self, clf, na_default=0):
        self.clf = clf
        self.na_default = na_default
        
    def fit(self, X, y=None):
        mask = np.isnan(X).any(axis=1)
        
        X_non_na=X[~mask]
        
        if y is not None:
            y_non_na = y[~mask]
        else:
            y_non_na = None
        
        self.clf.fit(X_non_na, y_non_na)
        
        return self
        
    def predict(self, X):
        mask = np.isnan(X).any(axis=1)
        
        X_non_na=X[~mask]
        
        y_pred = self.clf.predict(X_non_na)
        
        y=np.full(X.shape[0], self.na_default)
        y[~mask] = y_pred
        
        return y
    
    
    
class FilterFirstCol(BaseEstimator, ClassifierMixin):
    """
    Classifier wrapper which filters on the first column of X and fit/predicts only on filtered X
    At the same time the first column is dropped for the wrapped classifier
    The rows which are filtered out get `default` as their prediction
    """
    def __init__(self, clf, min_value, default):
        self.clf = clf
        self.min_value = min_value
        self.default = default
        
    def fit(self, X, y=None):
        select = X[:,0] > self.min_value
        
        X_fit = X[select, 1:]
        
        if y is not None:
            y_fit = y[select]
        else:
            y_fit = None
        
        self.clf.fit(X_fit, y_fit)
        
        return self
        
    def predict(self, X):
        select = X[:,0] > self.min_value
        
        X_pred=X[select, 1:]
        y_pred = self.clf.predict(X_pred)
        
        y=np.full(X.shape[0], self.default)
        y[select] = y_pred
        
        return y


def threshold_agg(series, window="10min", threshold=3, name=None):
    """
    Filters a series with 0s and 1s to those cases where the 1s occur at least `threshold` time in the last `window`
    """
    if name is None:
        name = f"agg_{window}_ge_{threshold}_{series.name}"

    agg = series.rolling(window).sum()

    result = (agg.ge(threshold) & series.eq(1)).astype(int).rename(name)

    return result


def add_agg_anom_col(
    dd,
    model_colname: ModelSpec2feats,
    *,
    min_anom_count=1,
    anom_count_window="10min",
    fit_sample_num=None,
):
    """
    Will add a new column to the data set where a model specified in `model_colname` will be fitted
    and will predict a new column which gets aggregated in time with `threshold_agg` to filter for cases
    where the anomaly occured more often within a time window
    
    dd: a new column will be added (modified inplace)
    """
 
    model = model_colname.model
    col_name = model_colname.col_name

    if fit_sample_num is None:
        X = dd
    else:
        X = dd.sample(fit_sample_num)
        
    model.fit(X)

    X_predict = dd
    y_predict = model.predict(X_predict)

    y_predict_series = pd.Series(y_predict, index=X_predict.index)
    y_agg_threshold = threshold_agg(
        y_predict_series,
        window=anom_count_window,
        threshold=min_anom_count,
        name=col_name,
    )

    dd[col_name] = y_agg_threshold
    dd[col_name] = dd[col_name].fillna(0).astype(int)
    
    return col_name


def multialarm_score(dd, models, *, anom_weights, agg_param=None, anom_col_name="multialarm_score", progress_bar=True):
    """
    Applies multiple models to get individual anomaly scores, which then get aggregated by a weighted sum to a total multi-alarm score.
    Will add the individual alarm columns to the data and also the total `anom_col_name`.
    """
    if agg_param is None:
        agg_param = {}
        
    if progress_bar:
        from tqdm import tqdm
        models = tqdm(models)

    anom_cols = []
    
    for model in models:                 # Cycle through all models (different feature pairs and possible different model parameters)
        col_name = add_agg_anom_col(     # New anomaly columns to DataFrame
            dd,
            model,
            **agg_param,
        )

        anom_cols.append(col_name)
        
    multialarm_score = sum(weight * dd[col] for col, weight in anom_weights.items()).rename(anom_col_name)
    
    dd[anom_col_name] = multialarm_score
    
    return anom_cols
