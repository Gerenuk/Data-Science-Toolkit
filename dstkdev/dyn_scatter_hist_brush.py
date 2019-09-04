%%output size=150
%%opts Points [tools=["lasso_select", "box_select"]]


# To display --> HIER EINGEBEN WELCHER SCATTERPLOT DIE BASIS IST
col1 = cm[66]
col2 = cm[202]

##############################
def outlier_dynhist(points, color, col):
    def hist(index):
        if index:
            selected = points.iloc[index]
        else:
            selected = points          

        return hv.Histogram(
            np.histogram(selected[col], bins="doane", density=True), kdims=col
        ).opts(fill_color=color, alpha=0.3, framewise=True, axiswise=True)

    selection = hv.streams.Selection1D(source=points)

    return hv.DynamicMap(hist, streams=[selection])


def plot_outlier(data, col1, col2, hist_cols):
    points0 = hv.Points(data.query("outlier==0"), kdims=[col1, col2])
    points1 = hv.Points(data.query("outlier==1"), kdims=[col1, col2])

    plot = hv.Layout(
        [points0.opts(color=colors[0]) * points1.opts(color=colors[1])]
        + [
            outlier_dynhist(points0, colors[0], col)
            * outlier_dynhist(points1, colors[1], col)
            for col in hist_cols
        ]
    )

    return plot