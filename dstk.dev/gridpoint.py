from math import cos, sin
import numpy as np


def circle_num(x):
    angle = 2 * pi * x
    return cos(angle), sin(angle)


dim = 3
dirdim = 6
# dim=4
# dirdim=int(2*dim/(dim-2))


class Point:
    """
    2D points represented in coordinates which are helpful for grids (e.g. hexagonal grid).
    The coordinates axes are exp(2*pi*I*k/D) where D is the dimensions (triangular 3, square 4)
    The sum of all coordinate coefficients is constraint to be zero (or should be normalized to sum 0 for uniqueness of representation)
    The grid points are best represented as integer coordinate coefficients (e.g. (1,-1,0)).
    Due to the constraint, (1,0,0,...) is not a normalized grid point.

    The distance metric measures a directed distance along the bisector.
    For triangles and squares the sum of all distances from a point to the nearest cell corners is constant (e.g. Viviani's theorem).
    """

    dim = dim
    dirdim = dirdim

    base_trans = np.array([circle_num(i / dim) for i in range(dim)])
    dist_trans = np.array(
        [
            [circle_num(i / dim - j / dirdim)[0] for j in range(dirdim)]
            for i in range(dim)
        ]
    )
    dist_trans /= max(
        dist_trans[0, :] - dist_trans[1, :]
    )  # normalize on point(1,-1,0,..)

    def __init__(self, *coefs):
        if len(coefs) != dim:
            raise ValueError(f"Expected {dim} coefficients but got {len(coefs)}")

        if not isclose(sum(coefs), 0, abs_tol=1e-10):  # allow for norm=True parameter?
            raise ValueError(
                f"Coefficients are constraint to sum zero. But received coefficients sum to {sum(coefs)}"
            )

        self.coefs = np.array(coefs)

    @classmethod
    def from_cartesian(cls, x, y):
        coefs = np.array([x, y]) @ self.base_trans.T
        return cls(*coefs)

    def to_cartesian(self):
        return 2 / self.dim * (self.coefs @ self.base_trans)

    def dist_to(self, other):
        """
        Manhattan-like distance which measures along the direction of the bisector.
        """
        coefs_diff = other.coefs - self.coefs
        dist = max(
            coefs_diff @ self.dist_trans
        )  # the largest distance is always the one for the relevant cell which contains the point
        return dist

    def grid_shares(self):  # this version works only for dim=3
        """
        Returns [(share_of_point, point), ...]
        Number of points is equal to number of dimensions
        Unless it is already a perfect grid point, in which case only [(1, grid_point)] is returned
        """
        floor_coefs = np.floor(self.coefs).astype(int)  # int for short str repr

        if (floor_coefs == self.coefs).all():
            add_coefs = [
                (0, 0, 0)
            ]  # will result in only one point when it already is a perfect grid point
        elif sum(floor_coefs) % 2 == 0:
            add_coefs = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
        else:
            add_coefs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        points = [Point(coefs=floor_coefs + np.array(list(a))) for a in add_coefs]

        return [(1 - point.dist_to(self), point) for point in points]

    def id(self):
        return ",".join(map(str, self.coefs[:-1]))  # skip last since sum zero

    def __eq__(self, other):
        return self.coefs == other.coefs

    def __hash__(self):
        return hash(tuple(self.coefs))

    def __repr__(self):
        return f"P({', '.join(map(str, self.coefs))})"

    @classmethod
    def zero(cls):
        return cls(coefs=[0] * dim)


class GeoGridScaler:
    """
    Takes longitude and latitude and rescales to a locally square grid with some edge length.
    """
    def __init__(self, scale):
        self.scale = scale

    def to_grid(self, lat, lon):
        x = lat / self.scale
        y = lon * cos(2 * pi / 360 * lat) / self.scale
        return x, y

    def to_geo(self, x, y):
        lat = x * self.scale
        lon = y * self.scale / cos(2 * pi / 360 * lat)
        return lat, lon

