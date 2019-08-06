from scipy.cluster import hierarchy


def plot_cluster_map(
    dataframe,
    linkage_method="complete",
    metric="correlation",
    figsize=None,
    dendrogram_kwargs=None,
    save_filename=None,
):
    """
    :param dataframe: Features are in columns -> will cluster features (no data instances)
    :param linkage_methods: single, complete, average, weighted; only with Euclidean metric: centroid, median, ward

    Clustering: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Plotting: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    Distances: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    """

    if dendrogram_kwargs is None:
        dendrogram_kwargs = {}

    cluster_dataframe = dataframe.T

    linkage_data = hierarchy.linkage(cluster_dataframe, method=linkage_method, metric=metric)

    if figsize is None:
        figsize = (10, len(cluster_dataframe)*0.2)
    
    fig, ax = plt.subplots(figsize=figsize)

    labels = d.index

    hierarchy.dendrogram(
        linkage_data,
        orientation="right",
        labels=labels,
        ax=ax,
        distance_sort=True,
        **dendrogram_kwargs,
    )

    ax.tick_params(axis="y", which="major", labelsize=8)
    
    ax.grid(True)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.xaxis.grid(True, which="minor", linestyle="--")

    if save_filename is not None:
        fig.savefig(save_filename, bbox_inches="tight")

    return ax, linkage_data
