import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from itertools import islice, chain, repeat
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator, train_test_split
from sklearn.utils.validation import _num_samples, indexable
import colorful
from functools import partial

colorful.use_true_colors()


def format_if_number(x, color=None, format="{:g}"):
    if isinstance(x, (int, float)):
        text = format.format(x)
    else:
        text = x

    if color is not None:
        text = color(text)

    return text


color_score = partial(format_if_number, color=colorful.violet)
color_number = partial(format_if_number, color=colorful.cornflowerBlue, format="{}")
color_param_val = partial(format_if_number, color=colorful.deepSkyBlue)
color_param_name = partial(format_if_number, color=colorful.limeGreen, format="{}")


def featimp(clf, df):
    return pd.Series(clf.feature_importances_, index=df.columns).sort_values(
        ascending=False
    )


def earlystop(
    clf,
    X,
    y,
    *,
    eval_metric=None,
    early_stopping_rounds=100,
    test_size=0.1,
    num_feat_imps=5,
    shuffle=False,
    **fit_params,
):
    X_train, X_stop, y_train, y_stop = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )

    clf.fit(
        X_train,
        y_train,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_stop, y_stop)],
        eval_metric=eval_metric,
        **fit_params,
    )

    infos = []
    if hasattr(clf, "best_iteration_") and clf.best_iteration_ is not None:
        infos.append(f"Best iter {clf.best_iteration_}")

        if hasattr(clf, "best_score_") and clf.best_score_:   # this variable is meant to come only from early stoppers
            if isinstance(clf.best_score_, dict):
                best_score_str = ", ".join(
                    (f"{set_name}(" if len(clf.best_score_) > 1 else "")
                    + ", ".join(
                        f"{score_name}={score:g}" for score_name, score in scores.items()
                    )
                    + (")" if len(clf.best_score_) > 1 else "")
                    for set_name, scores in clf.best_score_.items()
                )
            else:
                best_score_str = format_if_number(clf.best_score_)    # usually should not happen

            infos.append(f"Stop scores {best_score_str}")

    if hasattr(clf, "feature_importances_"):
        feat_imps = sorted(zip(clf.feature_importances_, X.columns), reverse=True)
        infos.append(
            "Top feat: "
            + " · ".join(feat for _score, feat in feat_imps[:num_feat_imps])
        )
    print("\n".join(infos))


class Mesh:
    def __init__(self, x, y, num_points):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        self.xx, self.yy = np.meshgrid(
            np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points)
        )
        self.mesh_shape = self.xx.shape

    @property
    def X(self):
        return np.c_[self.xx.ravel(), self.yy.ravel()]

    def mesh_reshape(self, x):
        return x.reshape(self.mesh_shape)


def plot_mesh_predict_contour(clf, x, y, ax=None, num_points=100, **contour_params):
    if ax is None:
        ax = plt.gca()

    mesh = Mesh(x, y, num_points)
    X = mesh.X
    z = clf.predict(X)
    zz = mesh.mesh_reshape(z)
    ax.contour(mesh.xx, mesh.yy, zz, **contour_params)


def cond_entropy(dd, y_cols, x_cols):
    counts = (
        dd.groupby(y_cols + x_cols)
        .size()
        .to_frame("xy")
        .reset_index()
        .merge(dd.groupby(x_cols).size().to_frame("x").reset_index())
    )
    return entropy(counts["xy"], counts["x"])


def baserepr(num, base):
    (d, m) = divmod(num, base)
    if d > 0:
        return baserepr(d, base) + (m,)
    return (m,)


def shuffle_maps(N, base, noise=0.1):
    reprs = [baserepr(i, base) for i in range(N)]

    len_repr = max(map(len, reprs))
    mappings = np.array(
        [
            list(islice(chain(reversed(a_repr), repeat(0)), len_repr))
            for a_repr in reprs
        ],
        dtype="float32",
    )

    mappings += np.random.normal(scale=noise, size=mappings.shape)

    return mappings


class DigitCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Like binary encoding, but with arbitrary base and a noise term.
    Does not treat NaN
    """

    def __init__(self, base=3, noise=0.1):
        self.base = base
        self.noise = noise
        self.mappings = []

    def fit(self, X, y=None):
        """
        X should be 1D
        """
        np.random.seed(123)

        uniques = sorted(X.unique())  # may be random
        shuffle_maps_ = shuffle_maps(len(uniques), base=self.base, noise=self.noise)

        for i in range(shuffle_maps_.shape[1]):
            mapping = {val: shuffle_maps_[j, i] for j, val in enumerate(uniques)}
            self.mappings.append(mapping)

        return self

    def transform(self, X):
        """
        Assuming X is Series with .name
        """
        name = X.name

        result = pd.DataFrame(
            {
                f"digicat_{name}_{i}": X.map(mapping)
                for i, mapping in enumerate(self.mappings)
            },
            index=X.index,
        )
        return result


class FutureSplit:
    def __init__(self, test_size, n_reduce=0):
        self.test_size = test_size
        self.n_reduce = n_reduce

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)

        n_samples = _num_samples(X)
        train_size = round((1 - self.test_size) * n_samples)

        train_index = np.arange(train_size - self.n_reduce)
        test_index = np.arange(train_size, n_samples)

        yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def __repr__(self):
        return (
            f"FutureSplit(test_size={self.test_size}"
            + (f", n_reduce={self.n_reduce}" if self.n_reduce > 0 else "")
            + ")"
        )


class KFoldGap:
    """
    Like KFold (without shuffle), but also reduces the training indices by a fixed
    number n_reduce to create a gap between test and training
    """

    def __init__(self, n_splits, n_reduce):
        self.n_splits = n_splits
        self.n_reduce = n_reduce

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)

        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
        fold_sizes[: n_samples % n_splits] += 1

        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size

            test_index = indices[start:stop]

            start_pad = start - self.n_reduce
            stop_pad = stop + self.n_reduce

            if start_pad < 0:
                start_pad = 0

            if stop_pad > n_samples:
                stop_pad = n_samples

            block_index = indices[start_pad:stop_pad]

            block_mask = np.zeros(n_samples, dtype=np.bool)
            block_mask[block_index] = True
            train_index = indices[np.logical_not(block_mask)]

            yield train_index, test_index

            current = stop

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def __repr__(self):
        return f"KFoldGap({self.n_splits}, n_reduce={self.n_reduce})"


class StratifyGroup:
    """
    Will try to distribute equal `groups` into separate folds
    """

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)

        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        argsort = np.argsort(groups)

        for i in range(self.n_splits):
            train_index = indices[argsort[i :: self.n_splits]]

            train_mask = np.zeros(n_samples, dtype=np.bool)
            train_mask[train_index] = True

            test_index = indices[np.logical_not(train_mask)]

            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
        
    def __repr__(self):
        return f"StratifyGroup({self.n_splits})"

        
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


class TrainOnlyFold:
    def split(self, X, y=None, groups=None):
        yield np.arange(X.shape[0]), np.array([])

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def __repr__(self):
        return "TrainOnlyFold()"
