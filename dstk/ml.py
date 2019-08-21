import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples, indexable

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
    counts=dd.groupby(y_cols+x_cols).size().to_frame("xy").reset_index().merge(dd.groupby(x_cols).size().to_frame("x").reset_index())
    return entropy(counts["xy"], counts["x"])


def baserepr(num, base):
    (d, m) = divmod(num, base)
    if d > 0:
        return baserepr(d, base) + (m,)
    return (m,)


def shuffle_maps(N, base, noise=0.1):
    reprs = [baserepr(i, base) for i in range(N)]

    len_repr = max(map(len, reprs))
    mappings = np.array([list(islice(chain(reversed(a_repr), repeat(0)), len_repr)) for a_repr in reprs], dtype="float32")

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
        self.shuffle_maps_ = None
        self.uniques_ = []

    def fit(self, X, y=None):
        """
        X should be 1D
        """
        self.uniques_ = {x: i for i, x in enumerate(set(X))}  # may be random
        self.shuffle_maps_ = shuffle_maps(len(self.uniques_), base=self.base, noise=self.noise)

        return self

    def transform(self, X, name=None):
        """
        Assuming X is Series with .name
        """
        if name is None:
            name = X.name

        result = pd.DataFrame(
            [
                [
                    self.shuffle_maps_[self.uniques_[x], i]
                    for i in range(self.shuffle_maps_.shape[1])
                ]
                for x in X
            ],
            columns=[f"digicat_{name}_{i}" for i in range(self.shuffle_maps_.shape[1])]
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


class StratifyGroup(BaseCrossValidator):
    """
    Will try to distribute equal `groups` into separate folds
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def _iter_test_indices(self, X, y=None, groups=None):
        """
        groups needs to be sortable
        """
        n_samples = _num_samples(X)

        n_splits = self.n_splits

        argsort = np.argsort(groups)

        for i in range(n_splits):
            yield argsort[i::n_splits]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
