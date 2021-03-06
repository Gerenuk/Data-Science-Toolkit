class Mesh:
    """
    Helper function to plot the contours of a Sklearn model
    """
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
    """
    Plot contours of classifier predictions by predicting for all points of a 2D grid
    """
    if ax is None:
        ax = plt.gca()
    
    mesh = Mesh(x, y, num_points)
    X = mesh.X
    z = clf.predict(X)
    zz = mesh.mesh_reshape(z)
    ax.contour(mesh.xx, mesh.yy, zz, **contour_params)
    
    
def plot_anom_contour(
    dd, model, mesh_model, feat1, feat2, *, num_top_anom_days=3
):
    """
    Extended plotting of contours of model predictions
    """
    plt.hexbin(feat1, feat2, data=dd, cmap="Greys", bins="log", label="")

    plot_mesh_predict_contour(
        mesh_model, dd[feat1], dd[feat2], colors="purple"
    )

    anom_day_counts = (
        dd
        .pipe(
            lambda d: d.groupby(d.index.date).apply(
                lambda g: anomaly_score(model, g)
            )
        )
        .sort_values(ascending=False)
    )

    for day in anom_day_counts.index[:num_top_anom_days]:
        dd_day = dd[str(day)]

        plt.plot(
            feat1,
            feat2,
            data=dd_day,
            marker=".",
            label=f"{day.strftime('%d.%m.%Y')} ({anom_day_counts[day]}x)",
            alpha=0.5,
        )

    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.legend()

    return anom_day_counts