import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from itertools import islice, chain, repeat
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import BaseCrossValidator, train_test_split
from sklearn.utils.validation import _num_samples, indexable, check_X_y
import colorful
from functools import partial
import scipy
from numba import jit

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
    groups=None,
    **fit_params,
):
    if groups is not None:
        X_train, X_stop, y_train, y_stop, groups_train, groups_stop = train_test_split(  #! groups_stop dodgy
            X, y, groups, test_size=test_size, shuffle=shuffle
        )
    else:
        X_train, X_stop, y_train, y_stop = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle
        )
        groups_train = None

    clf.fit(
        X_train,
        y_train,
        groups=groups_train,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_stop, y_stop)],
        eval_metric=eval_metric,
        **fit_params,
    )

    infos = []
    if hasattr(clf, "best_iteration_") and clf.best_iteration_ is not None:
        infos.append(f"Best iter {clf.best_iteration_}")

        if (
            hasattr(clf, "best_score_") and clf.best_score_
        ):  # this variable is meant to come only from early stoppers
            if isinstance(clf.best_score_, dict):
                best_score_str = ", ".join(
                    (f"{set_name}(" if len(clf.best_score_) > 1 else "")
                    + ", ".join(
                        f"{score_name}={score:g}"
                        for score_name, score in scores.items()
                    )
                    + (")" if len(clf.best_score_) > 1 else "")
                    for set_name, scores in clf.best_score_.items()
                )
            else:
                best_score_str = format_if_number(
                    clf.best_score_
                )  # usually should not happen

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

        X_non_na = X[~mask]

        if y is not None:
            y_non_na = y[~mask]
        else:
            y_non_na = None

        self.clf.fit(X_non_na, y_non_na)

        return self

    def predict(self, X):
        mask = np.isnan(X).any(axis=1)

        X_non_na = X[~mask]

        y_pred = self.clf.predict(X_non_na)

        y = np.full(X.shape[0], self.na_default)
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
        select = X[:, 0] > self.min_value

        X_fit = X[select, 1:]

        if y is not None:
            y_fit = y[select]
        else:
            y_fit = None

        self.clf.fit(X_fit, y_fit)

        return self

    def predict(self, X):
        select = X[:, 0] > self.min_value

        X_pred = X[select, 1:]
        y_pred = self.clf.predict(X_pred)

        y = np.full(X.shape[0], self.default)
        y[select] = y_pred

        return y


class TrainOnlyFold:
    def split(self, X, y=None, groups=None):
        yield np.arange(X.shape[0]), np.array([])

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def __repr__(self):
        return "TrainOnlyFold()"


class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """

    def __init__(self, sort_key=None, dtype="uint8"):
        self.sort_key = sort_key
        self.value_map_ = None
        self.dtype = dtype

    def fit(self, X, y=None):
        self.value_map_ = {
            val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))
        }
        return self

    def transform(self, X, y=None):
        values = X.map(self.value_map_)

        possible_values = sorted(self.value_map_.values())

        idx1 = []
        idx2 = []

        all_indices = np.arange(len(X))

        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))

        result = scipy.sparse.coo_matrix(
            ([1] * len(idx1), (idx1, idx2)),
            shape=(len(X), len(possible_values)),
            dtype=self.dtype,
        )

        return result


class ThermometerEncoder2(TransformerMixin):
    """
    Assumes all values are known at fit

    7x faster than ThermometerEncoder, but limited to at most 255 categories
    """

    def __init__(self, sort_key=None, dtype="uint8"):
        self.sort_key = sort_key
        self.value_map_ = None
        self.dtype = dtype

    def fit(self, X, y=None):
        self.value_map_ = {
            val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))
        }
        if len(self.value_map) > 255:  # Is it exactly 255?
            raise ValueError(
                "This ThermometerEncoder2 does not support more than 255 categories"
            )
        return self

    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        a = values.values.astype(np.uint8)  # Category limit!

        out = np.empty((len(a), 0), dtype=np.uint8)
        while a.any():
            block = np.fliplr(np.unpackbits((1 << a) - 1).reshape(-1, 8))
            out = np.concatenate([out, block], axis=1)
            a = np.where(a < 8, 0, a - 8)

        result = scipy.sparse.coo_matrix(out, dtype=self.dtype)

        return result


@jit(nopython=True)
def qwk_numba(a1, a2, num_classes):
    """
    Quadratic weighted Cohen kappa
    """
    counts = np.zeros((num_classes, num_classes))

    for x1, x2 in zip(a1, a2):
        counts[x1, x2] += 1

    hist1 = counts.sum(axis=0)
    hist2 = counts.sum(axis=1)

    w_obs = 0
    w_exp = 0
    for i in range(num_classes):
        for j in range(num_classes):
            w = (i - j) * (i - j)
            w_obs += w * counts[i, j]
            w_exp += w * hist1[i] * hist2[j]

    w_exp /= len(a1)

    return 1 - w_obs / w_exp


class ThresholdClf(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, thresholds):
        self.estimator = estimator
        self.thresholds = thresholds

    def fit(self, X, y, **kwargs):
        X, y = indexable(X, y)

        self.estimator.fit(X, y, **kwargs)

        return self

    def predict(self, X):
        pred = self.estimator.predict(X)

        y = np.digitize(pred, self.thresholds).astype(int)

        return y

    def __getattr__(self, name):
        if hasattr(self.estimator, name):
            return getattr(self.estimator, name)

        raise AttributeError


class Cascaded(BaseEstimator, ClassifierMixin):
    def __init__(self, first_estimator, first_class, final_estimator):
        """
        First estimator decides whether to output first_class (prediction 1) or
        run the second estimator to predict another class
        """
        self.first_estimator = first_estimator
        self.first_class = first_class
        self.final_estimator = final_estimator

    def fit(self, X, y):
        # X, y = check_X_y(X, y)

        y_first = (y == self.first_class).astype(int)

        self.first_estimator = clone(self.first_estimator)
        self.first_estimator.fit(X, y_first)

        select_final = y != self.first_class
        X_final = X[select_final]
        y_final = y[select_final]

        self.final_estimator = clone(self.final_estimator)
        self.final_estimator.fit(X_final, y_final)

    def predict(self, X):
        # X = check_array(X)

        y_first = self.first_estimator.predict(X)

        X_final = X[y_first == 0]

        y_final = self.final_estimator.predict(X_final)

        dtype = y_final.dtype
        result = np.empty(shape=X.shape[0], dtype=dtype)
        is_first = y_first.astype(bool)

        result[is_first] = self.first_class
        result[~is_first] = y_final

        return result


class BaggingClassifier:
    """
    Different from sklearn as it will keep "category" dtypes
    """

    def __init__(self, estimator, n_estimators, max_samples):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples

        self.seeds = range(self.n_estimators)

    def fit(self, X, y, do_oob_preds=False):
        estimators = [clone(self.estimator) for _ in range(self.n_estimators)]

        oob_ys = []
        oob_preds = []
        oob_idxs = []

        for seed, estimator in zip(self.seeds, estimators):
            num_select = int(X.shape[0] * self.max_samples)
            seed_selected_idxs = np.random.choice(X.shape[0], size=num_select)

            X_seed = X.iloc[seed_selected_idxs]
            y_seed = y.iloc[seed_selected_idxs]

            estimator.fit(X_seed, y_seed)

            if do_oob_preds:
                seed_oob_idxs = np.ones(X.shape[0], np.bool)
                seed_oob_idxs[seed_selected_idxs] = 0

                X_oob = X.loc[seed_oob_idxs]
                y_oob = y.loc[seed_oob_idxs]

                oob_pred = estimator.predict(X_oob)

                oob_ys.append(y_oob)
                oob_preds.append(oob_pred)
                oob_idxs.append(seed_oob_idxs)

        self.estimators_ = estimators

        if oob_preds:
            self.oob_ys_ = oob_ys
            self.oob_preds_ = oob_preds
            self.oob_idxs_ = oob_idxs

    def predict(self, X):
        part_preds = []
        for estimator in self.estimators_:
            part_pred = estimator.predict(X)
            part_preds.append(part_pred)

        part_preds = np.array(part_preds)

        mode_result = scipy.stats.mode(part_preds)

        self.part_preds_ = part_preds

        return mode_result.mode[0]
