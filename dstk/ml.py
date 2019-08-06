import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.base import TransformerMixin, BaseEstimator

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
