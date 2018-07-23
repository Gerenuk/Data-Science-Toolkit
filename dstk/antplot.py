from __future__ import division

import itertools
import inspect
from collections import Iterable

import bokeh
import bokeh.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial
from bokeh.models import HoverTool, BoxZoomTool, PanTool, ResetTool, SaveTool
from bokeh.models import Label, Range1d
from bokeh.plotting import ColumnDataSource, figure
from bokeh.palettes import all_palettes

__author__ = "Anton Suchaneck"
__email__ = "anton.suchaneck@ing-diba.de"

"""
TODO:
* check for NaN when sorting?
* tick formatter in coffeescript with percent

minauc:
* minauc could be generally on x=pos+a*tot, y=pos+b*tot (?) -> prec would still be slope, only recall would be less visible
* use ordered string values?
"""

heatmap_cmap = plt.cm.YlOrRd
DEFAULT_MAX_PTS = 10000
DEFAULT_BOKEH_TOOLS = [BoxZoomTool(), PanTool(), ResetTool(), SaveTool()]


def thin_out_idxs(x, max_len):
    """
    :param x: sorted numpy array
    :param max_len: number of elements to sample
    :return: indices of selected elements

    Returns at most max_len indices 0...len(x)-1 such that the subsampled x are evenly spread
    """
    if len(x) <= max_len:
        return np.arange(len(x))

    val_min = x[0]
    val_max = x[-1]

    estimates = np.arange(max_len) / (max_len - 1) * (val_max - val_min)

    idxs = np.searchsorted(x - val_min, estimates, side="left")

    return idxs


def hist(vals, **kwargs):
    """
    Use with s.pipe(hist)
    """
    import holoviews as hv
    return hv.Histogram(np.histogram(vals, bins="auto"), **kwargs)


def plot_roc(y_true, *y_preds, **kwargs):  # kwargs as hack for Python 2
    """
    :param y_true: list of true instance labels (0 or 1); should not include NaNs!
    :param y_preds: one or multiple lists of instance scores (e.g. from .predict_proba())
    :param names: names corresponding to y_preds to display in the legend; can be single string if only one y_pred given
    :param colors: colors to be used for y_preds (otherwise a default palette will be used)
    :param precs: list of precision iso-lines to display
    :param scores: list of score iso-lines to display (cost matrix: truepos=score, falsepos=-1, trueneg=0, falseneg=0)
    :param numpreds: list of prediction count iso-lines to show (a line for where specific number of pos labels)
    :param accs: list of accuracy iso-lines to show (acc=True will display just one line through (0,0) with the correct slope)
    :param fig: Bokeh figure to be used for plotting
    #:param threshold_color:
    :param max_pts: maximum number of points to show in plot
    :return: Bokeh figure

    Usage:

    >>> from individual.anton.antplot import plot_roc
    >>> from bokeh.plotting import output_notebook
    >>> output_notebook()
    >>> plot_roc(y_true, y_pred_rf, y_pred_logreg, accs=[0.8, 0.85], names=["RF", "LogReg"], precs=[0.9, 0.6], numpreds=[300]);

    Multiple curves with different y_true:

    >>> fig=figure()
    >>> plot_roc(y_test1, y_pred1, names="A", fig=fig)
    >>> plot_roc(y_test2, y_pred2, names="B", fig=fig)
    >>> show(fig)

    Display:
    Count end: Number of remaining points (not including current one)
    """
    from sklearn.metrics import roc_auc_score

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    y_preds = [y if isinstance(y, np.ndarray) else np.array(y) for y in y_preds]

    for name, y_vals in ([("y_true", y_true)] +
                         [("y_preds[{}]".format(i), y_pred) for i, y_pred in enumerate(y_preds)]
    ):
        if np.any(np.isnan(y_vals)):
            raise ValueError("Forbidden NaN values found in {}".format(name))

    forbidden_ytrue_vals = set(np.unique(y_true)) - {0, 1}
    if forbidden_ytrue_vals:
        raise ValueError(
            "y_true contains more than just 0 and 1: {}".format(", ".join(map(str, list(forbidden_ytrue_vals)[:10]))))

    # hack since Python 2 cannot do it
    names = kwargs.pop("names",
                       [""] if len(y_preds) == 1 else [str(i) for i in range(len(y_preds))])
    if isinstance(names, str) and len(y_preds) == 1:
        names = [names]

    colors = kwargs.pop("colors", [])
    precs = kwargs.pop("precs", [])
    scores = kwargs.pop("scores", [])
    numpreds = kwargs.pop("numpreds", [])
    accs = kwargs.pop("accs", None)
    fig = kwargs.pop("fig", None)
    large_scores_first = kwargs.pop("large_scores_first", True)
    max_pts = kwargs.pop("max_pts", DEFAULT_MAX_PTS)
    title = kwargs.pop("title", "")
    # threshold_color = kwargs.pop("threshold_color", False)

    if kwargs:
        raise ValueError("Unknown arguments to plot_roc: {}".format(sorted(kwargs.keys())))

    if names is None:
        names = []
    names = itertools.chain(names, itertools.repeat(None))

    numpos = np.sum(y_true)
    numneg = len(y_true) - numpos

    hover = HoverTool(tooltips=[("Recall", "@y"),
                                ("Fallout", "@x"),
                                ("Precision", "@precs"),
                                ("Accuracy", "@accs"),
                                ("Count", "@point_counts"),
                                ("Count end", "@remaining_counts"),
                                ("Threshold", "@thresholds"),
                                ])

    show = fig is None

    if fig is None:
        fig = figure()

    if not hasattr(fig, "colorcycle_"):
        fig.__dict__["colorcycle_"] = itertools.chain(colors, itertools.cycle(
            bokeh.palettes.Set1_9))  # hack; otherwise Bokeh complains

    if not any(isinstance(tool, HoverTool) for tool in fig.tools):
        fig.add_tools(hover)

    fig.title.text = title

    fig.line([0, 1], [0, 1], line_dash="dotted", line_color="black")  # Random diagonal

    for y_pred, name, color in zip(y_preds, names, fig.colorcycle_):
        idx_order = y_pred.argsort() if not large_scores_first else y_pred.argsort()[::-1]

        ordered_y_true = y_true[idx_order]
        cnt_ys = np.cumsum(ordered_y_true)
        cnt_xs = np.arange(1, len(cnt_ys) + 1) - cnt_ys
        thresholds = y_pred[idx_order]

        x = cnt_xs / cnt_xs[-1]
        y = cnt_ys / cnt_ys[-1]
        prec_vals = cnt_ys / (cnt_xs + cnt_ys)
        acc_vals = (cnt_ys + cnt_xs[-1] - cnt_xs) / len(y_pred)

        point_counts = np.arange(1, len(x) + 1)
        remaining_counts = len(x) - point_counts

        # Do actual ROC plot here
        idxs = thin_out_idxs(x, max_pts)
        x_plot = x[idxs]
        y_plot = y[idxs]
        thresholds_plot = thresholds[idxs]
        prec_plot = prec_vals[idxs]
        acc_plot = acc_vals[idxs]
        point_counts_plot = point_counts[idxs]
        remaining_counts_plot = remaining_counts[idxs]

        data = ColumnDataSource(data=dict(x=x_plot,
                                          y=y_plot,
                                          thresholds=thresholds_plot,
                                          point_counts=point_counts_plot,
                                          remaining_counts=remaining_counts_plot,
                                          precs=prec_plot,
                                          accs=acc_plot,
                                          ))

        name += " auc:{:.2f}".format(roc_auc_score(y_true, y_pred))  # calc yourself?

        fig.line("x", "y",
                 source=data,
                 line_width=2,
                 line_color=color,
                 # line_color="thresholds",
                 # color_mapper=LinearColorMapper(),
                 legend=name,
                 )

    fig.legend.location = "center_right"

    def put_label_on_line(x_end, **label_args):
        if x_end < 1:
            fig.add_layout(
                Label(x=x_end, y=1 - 0.025, **label_args))
        else:
            fig.add_layout(Label(x=1, y=1 / x_end, text_align="right", **label_args))

    for prec in precs:
        x_end = (1 - prec) / prec * numneg / numpos
        fig.line([0, x_end], [0, 1], line_dash="dashed", line_color="orange")
        put_label_on_line(x_end,
                          text="prec" + str(prec),
                          text_color="orange",
                          text_font_size="8pt")

    if accs is True:
        x_end = numneg / numpos
        color = "magenta"
        fig.line([0, x_end], [0, 1], line_dash="dashed", line_color=color,
                 legend="Accuracy")
    elif accs:
        for acc in accs:
            y_beg = (acc * (numpos + numneg) - numneg) / numpos
            slope = numneg / numpos
            x_end = (1 - y_beg) / slope
            color = "magenta"
            fig.line([0, x_end], [y_beg, 1], line_color=color, line_dash="dashed")
            put_label_on_line(x_end,
                              text="acc" + str(acc),
                              text_color=color,
                              text_font_size="8pt",
                              )

    for score in scores:
        x_end = 1 / score * numneg / numpos
        color = "green"
        fig.line([0, x_end], [0, 1], line_dash="dashed", line_color=color)
        put_label_on_line(x_end,
                          text="scr" + str(score),
                          text_color=color,
                          text_font_size="8pt")

    for num_pred_pos_el in numpreds:
        y_beg = num_pred_pos_el / numpos
        x_end = num_pred_pos_el / numneg
        fig.line([0, x_end], [y_beg, 0], line_dash="dashed", line_color="grey")
        fig.add_layout(
            Label(x=x_end, y=0, text=("#pred" + str(num_pred_pos_el)), text_color="grey", text_font_size="8pt"))

    fig.x_range = Range1d(0, 1)
    fig.y_range = Range1d(0, 1)
    fig.xaxis.axis_label = "Fallout"
    fig.yaxis.axis_label = "Recall"

    if show:
        bokeh.io.show(fig)  # needed or later or show=True parameter?


def plot_cumu(data, weights=None, source=None,
              data_label=None, weights_label=None,
              tooltip_cols=tuple(),
              norm_weights=False,
              max_pts=DEFAULT_MAX_PTS, show=True, **plotargs):
    """
    Sorts by data and cumulatively sums weights.

    Specify

    * `data` (and `weights`) as vectors or
    * `source` as DataFrame and `data` (and `weights`) as column names

    :param data: data column or column name (if `source` given)
    :param weights: weights column or column name (if `source` given); default all 1's
    :param source: Pandas DataFrame for data (if column names specified)
    :param data_label: Label to use for plotting
    :param weights_label: Label to use for plotting
    :param tooltop_cols: List of additional columns to show in hover tooltip (needs `source`)
    :param norm_weights: Whether cumulated weights should be normalized to 1
    :param max_pts: Maximum number of points to show
    :param show: Whether Bokeh figure show also be shown right away
    :param plotargs: Parameters passed to Bokeh `.line()`
    :return: Bokeh figure

    Usage:

    >>> from individual.anton.antplot import plot_cumu
    >>> from bokeh.plotting import output_notebook
    >>> output_notebook()
    >>> plot_cumu("datacol", source=df)
    >>> plot_cumu(df["datacol"])
    """
    if source is not None:
        df = source.copy()
        if weights is None:
            df["weights"] = [1] * len(df)  # what if exists?
            weights = "weights"
            if weights_label is None:
                weights_label = "Count"
    else:
        if weights is None:
            weights = [1] * len(data)
            if weights_label is None:
                weights_label = "Cumul. Count"
        df = pd.DataFrame({"data": data, "weights": weights})
        data = "data"
        weights = "weights"

    if data_label is None:
        data_label = data

    if weights_label is None:
        weights_label = "Cumul. " + weights

    df.sort_values(data, ascending=True, inplace=True)
    df[weights + "_cumul"] = df[weights].cumsum()

    if norm_weights:
        df[weights + "_cumul"] = df[weights + "_cumul"] / df[weights + "_cumul"].max()
        weights_label = "Norm. " + weights_label

    # cntr[ceil((data_el - data_min) / data_max_min * max_pts) if data_el > data_min else 1] += weight

    tooltip_col_map = {col_name: "col" + str(i) for i, col_name in enumerate(tooltip_cols)}

    tooltips = [(weights_label, "@y"),
                (data_label, "@x"),
                ] + [
                   (col_name, "@" + tooltip_col_map[col_name]) for col_name in tooltip_cols
               ]

    hover = HoverTool(tooltips=tooltips)

    fig = figure(x_axis_label=data_label,
                 y_axis_label=weights_label,
                 tools=[hover] + DEFAULT_BOKEH_TOOLS)

    idxs = thin_out_idxs(df[data], max_pts)
    x_plot_thin = df[data][idxs]
    data_thin = df.iloc[idxs]

    col_data = dict(x=x_plot_thin, y=data_thin[weights + "_cumul"])

    for col_name in tooltip_cols:
        col_data[tooltip_col_map[col_name]] = data_thin[col_name]

    data = ColumnDataSource(data=col_data)
    fig.line("x", "y", source=data, **plotargs)

    if show:
        bokeh.io.show(fig)

    return fig


def plot_conc(data, weights=None, source=None,
              data_label=None, weights_label=None,
              norm_data=False, norm_weights=False,
              tooltip_cols=tuple(),
              show=True, largest_first=True, max_pts=DEFAULT_MAX_PTS,
              **plotargs):
    """
    Will sort by data and accumulate on both data and weights

    Specify

    * `data` (and `weights`) as vectors or
    * `source` as DataFrame and `data` (and `weights`) as column names

    :param data: data column or column name (if `source` given)
    :param weights: weights column or column name (if `source` given); default all 1's
    :param source: Pandas DataFrame for data (if column names specified)
    :param data_label: Label to use for plotting
    :param weights_label: Label to use for plotting
    :param tooltop_cols: List of additional columns to show in hover tooltip (needs `source`)
    :param norm_weights: Whether cumulated data should be normalized to 1
    :param norm_weights: Whether cumulated weights should be normalized to 1
    :param max_pts: Maximum number of points to show
    :param show: Whether Bokeh figure show also be shown right away
    :param plotargs: Parameters passed to Bokeh `.line()`
    :return: Bokeh figure

    Usage:

    >>> from individual.anton.antplot import plot_conc
    >>> from bokeh.plotting import output_notebook
    >>> output_notebook()
    >>> plot_conc("data", source=df)
    >>> plot_conc(df["data"])
    >>> plot_conc(..., tooltop_cols=["a1", "a2"])
    """
    if source is not None:
        df = source.copy()
        if weights is None:
            df["weights"] = [1] * len(df)  # what if exists?
            weights = "weights"
            if weights_label is None:
                weights_label = "Count"
    else:
        if weights is None:
            weights = [1] * len(data)
        df = pd.DataFrame({"data": data, "weights": weights})
        data = "data"
        weights = "weights"

    if data_label is None:
        data_label = "Cumul. " + data

    if weights_label is None:
        weights_label = "Cumul. " + weights

    df.sort_values(data, ascending=False, inplace=True)
    df[data + "_cumul"] = df[data].cumsum()
    df[weights + "_cumul"] = df[weights].cumsum()

    # fig.yaxis.formatter = FuncTickFormatter(code=
    # """
    # function (tick) {{
    #    return tick + " (" + (tick / {}).toFixed(0) + "%)"
    # }};
    # """.format(max(y_plot)))

    if norm_weights:
        df[weights + "_cumul"] = df[weights + "_cumul"] / df[weights + "_cumul"].max()
        weights_label = "Norm. " + weights_label

    if norm_data:
        df[data + "_cumul"] = df[data + "_cumul"] / df[data + "_cumul"].max()
        data_label = "Norm. " + data_label

    tooltip_col_map = {col_name: "col" + str(i) for i, col_name in enumerate(tooltip_cols)}

    tooltips = [(data_label, "@y"),
                (weights_label, "@x"),
                ] + [
                   (col_name, "@" + tooltip_col_map[col_name]) for col_name in tooltip_cols
               ]

    hover = HoverTool(tooltips=tooltips)

    fig = figure(x_axis_label=weights_label,
                 y_axis_label=data_label,
                 tools=[hover] + DEFAULT_BOKEH_TOOLS)

    idxs = thin_out_idxs(df[weights + "_cumul"], max_pts)
    x_plot_thin = df[weights + "_cumul"][idxs]
    data_thin = df.iloc[idxs]

    col_data = dict(x=x_plot_thin, y=data_thin[data + "_cumul"])

    for col_name in tooltip_cols:
        col_data[tooltip_col_map[col_name]] = data_thin[col_name]

    data = ColumnDataSource(data=col_data)
    fig.line("x", "y", source=data, **plotargs)

    if show:
        bokeh.io.show(fig)

    return fig


def heatmap(x, y, xlabel=None, ylabel=None, title=None, log=False, **kwargs):
    ax = plt.gca()

    cmap = heatmap_cmap
    # cmap.set_under(color="white")

    ax.hexbin(x, y, cmap=cmap, mincnt=1, bins="log" if log else None, **kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return ax


def conf_mat(y_true, y_pred, reverse=True):
    """
    :param y_true: True labels (0 or 1)
    :param y_pred: Predicted probabilities (0 to 1)
    :return: HTML table (ipy_table) which will be displayed in Jupyter

    Needs `pip install ipy_table`

    Usage:

    >>> conf_mat(y_true, y_pred)
    """
    from ipy_table import make_table, set_cell_style, apply_theme

    df = pd.DataFrame({"pred": y_pred, "true": y_true})
    cm = pd.pivot_table(df, index="true", columns="pred", aggfunc=len).fillna(0).astype(int)
    classes = sorted(cm.index | cm.columns, reverse=reverse)
    cm = cm.reindex(classes, classes, fill_value=0)

    precs = pd.Series([cm.ix[c, c] / cm.ix[c, :].sum() for c in classes], index=classes)
    recall = pd.Series([cm.ix[c, c] / cm.ix[:, c].sum() for c in classes], index=classes)
    accuracy = sum(cm.ix[c, c] for c in classes) / cm.sum().sum()

    tot_true = cm.sum(axis=0)
    tot_pred = cm.sum(axis=1)

    table_data = [["Pred ->"] + classes + ["Total", "Recall"]]
    total = cm.sum().sum()

    for class_j, field_type_j in [(c, "class") for c in classes] + [(None, "total"), (None, "precision")]:
        table_row = []

        for class_i, field_type_i in [(None, "name")] + [(c, "class") for c in classes] + [(None, "total"),
                                                                                           (None, "recall")]:
            val = {("name", "class"): class_j,
                   ("name", "precision"): "Precision",
                   ("name", "total"): "Total",
                   ("class", "class"): cm.ix[class_i, class_j],
                   ("precision", "class"): "{:.1%}".format(precs[class_i]),
                   ("precision", "total"): "",
                   ("precision", "recall"): "{:.1%}".format(accuracy),
                   ("recall", "class"): "{:.1%}".format(recall[class_j]),
                   ("recall", "total"): "",
                   ("total", "class"): tot_true[class_j],
                   ("class", "total"): tot_pred[class_i],
                   ("total", "total"): total,
                   }[field_type_i, field_type_j]

            table_row.append(val)

        table_data.append(table_row)

    tab = make_table(table_data)
    apply_theme("basic_both")
    num_classes = len(classes)
    set_cell_style(1, 1, thick_border="left, top")
    set_cell_style(1, num_classes, thick_border="top,right")
    set_cell_style(num_classes, 1, thick_border="left,bottom")
    set_cell_style(num_classes, num_classes, thick_border="bottom,right")
    for i in range(2, num_classes):
        set_cell_style(i, 1, thick_border="left")
        set_cell_style(i, num_classes, thick_border="right")
        set_cell_style(1, i, thick_border="top")
        set_cell_style(num_classes, i, thick_border="bottom")
    return tab


def minauc(y_true, y_score, first="high", title="", show=True, trim_to=None):
    """
    Values are <=score (or >=score for 'inv' values)
    Hence N + invN = len(y_true)+(curval) since the current point belongs to both

    Note that it may smear cutoff when overall minauc is small throughout

    trim_to: Limit plot (x-axis) to a range to show roughly that ratio of the data (plus extension left/right)
    """
    assert first in ["high", "low"], 'Parameter `first` can only be "high" or "low"'
    assert len(y_true) == len(y_score), "y_true length {} and y_score length {} are different size".format(len(y_true),
                                                                                                           len(y_score))

    if hasattr(y_score, "name"):
        score_name = y_score.name
        if title is None:
            title = score_name
    else:
        score_name = "Score"

    df = pd.DataFrame({"y_true": y_true, "y_score": y_score, "count": 1})  # using pandas instead of numpy for groupby

    N = len(df)
    cnts = df.groupby("y_true").size().to_dict()
    assert cnts.keys() == {0, 1}, "y_true should only contain 0 or 1, however {} different values found".format(
        len(cnts))

    df_grouped = df.groupby("y_score").sum().reset_index()
    df_grouped = df_grouped.sort_values("y_score", ascending=(first == "low"))

    df_grouped["cumu_y_true"] = df_grouped["y_true"].cumsum()
    df_grouped["cumu_count"] = df_grouped["count"].cumsum()

    recfalls = (df_grouped["cumu_y_true"] - df_grouped["cumu_count"] * cnts[1] / N) * N / cnts[0] / cnts[1]
    aucs = (recfalls + 1) / 2

    pos_max = aucs.argmax()
    title += " {:.3f} @ {:.4g}".format(aucs.iloc[pos_max], df_grouped.iloc[pos_max]["y_score"])

    fig = figure(x_axis_label="N",
                 y_axis_label="minAUC",
                 title=title,
                 )

    fig.title.align = "center"

    data = ColumnDataSource({"n": df_grouped["cumu_count"],
                             "score": df_grouped["y_score"],
                             "recall": df_grouped["cumu_count"] / N,
                             "precision": df_grouped["cumu_y_true"] / df_grouped["cumu_count"],
                             "minauc": aucs,
                             "inv_n": N + df_grouped["count"] - df_grouped["cumu_count"],
                             "inv_recall": (N + df_grouped["count"] - df_grouped["cumu_count"]) / N,
                             "inv_precision": (cnts[1] + df_grouped["y_true"] - df_grouped["cumu_y_true"]) / (
                                     N + df_grouped["count"] - df_grouped["cumu_count"]),
                             })

    fig.line("n", "minauc", source=data)

    fig.add_tools(HoverTool(tooltips=[("minAUC", "@minauc"),
                                      (score_name, "@score"),
                                      ("N", "@n"),
                                      ("Recall", "@recall"),
                                      ("Precision", "@precision"),
                                      ("invN", "@inv_n"),
                                      ("invRecall", "@inv_recall"),
                                      ("invPrecision", "@inv_precision"),
                                      ],
                            attachment="vertical",
                            line_policy="nearest",
                            ))

    if trim_to is not None:
        num_1s = cnts[1]
        tail_part = (1 - trim_to) / 2

        idx_low0 = df_grouped["cumu_y_true"].searchsorted(num_1s * tail_part)[0]
        idx_high0 = df_grouped["cumu_y_true"].searchsorted(num_1s * (1 - tail_part))[0]
        idx_diff = idx_high0 - idx_low0

        # Extend for value to show double the range
        idx_low = idx_low0 - idx_diff // 2
        idx_high = idx_high0 + idx_diff // 2

        if idx_low < 0:
            idx_low = 0
        if idx_high > len(df_grouped) - 1:
            idx_high = len(df_grouped) - 1

        x_low = df_grouped.iloc[idx_low]["cumu_count"]
        x_high = df_grouped.iloc[idx_high]["cumu_count"]

        fig.x_range = Range1d(x_low, x_high)

    if show:
        bokeh.io.show(fig)

    return fig


class bofig:
    """
    Replacement for Bokeh.figure which adds:
    * tooltips per plot (e.g. tooltips = ["x", "a"])
    * color cycling
    * functions or raw data series instead of just column names (when source is given)
    * tooltips and automatic naming when raw data series are given
    * thinning out to max number of points (by args[0])
    """

    def __init__(self, *args, **kwargs):
        self.fig = figure(*args, **kwargs)
        self.colors = itertools.cycle(all_palettes["Set1"][9])

    def __getattr__(self, name):
        return partial(self._call_fig, getattr(self.fig, name))

    def _call_fig(self, plot_func, *args, max_pts=DEFAULT_MAX_PTS, tooltips=None, **kwargs):
        """
        assumes DataFrame for "source"
        """
        if "source" not in kwargs:
            kwargs["source"] = pd.DataFrame()

        if plot_func.__name__ == "scatter":
            param_names = ["x", "y", "size", "marker", "color"]
        else:
            param_names = list(p.name for p in itertools.takewhile(lambda p: p.kind == p.POSITIONAL_OR_KEYWORD,
                                                                   inspect.signature(plot_func).parameters.values()))

        args = list(args)  # to modify

        for i, (param_name, value) in enumerate(zip(param_names, args)):
            if param_name in {"x", "y"}:
                if (not isinstance(value, str) and
                        isinstance(value, Iterable)):
                    args[i] = param_name
                    kwargs["source"][param_name] = value
                elif callable(value) and "source" in kwargs:
                    data_series = value(kwargs["source"])
                    data_name = (data_series.name if hasattr(data_series, "name") and data_series.name is not None
                    else param_name)
                    args[i] = data_name
                    kwargs["source"][data_name] = data_series

        args = tuple(args)

        if "color" not in kwargs:
            kwargs["color"] = next(self.colors)

        if max_pts is not None and len(kwargs["source"]) > max_pts:
            source = kwargs["source"]
            first_arg = args[0]
            xs = source[first_arg] if isinstance(first_arg, str) else first_arg
            kwargs["source"] = source.iloc[thin_out_idxs(xs, max_pts)]

        if tooltips is None:
            tooltips = sorted(kwargs["source"].columns)

        kwargs["source"] = ColumnDataSource(kwargs["source"])  # otherwise legend=".." fails due to bool(df)

        plot = plot_func(*args, **kwargs)

        if tooltips:  # could be empty
            tooltips = [t if not isinstance(t, str) else (t, "@{}".format(t)) for t in tooltips]
            self.fig.add_tools(HoverTool(tooltips=tooltips, renderers=[plot]))

    def show(self):
        bokeh.io.show(self.fig)


if __name__ == '__main__':
    import random

    N = 1000
    y_true = [random.choice([0, 1]) for _ in range(N)]
    y_pred = [random.random() for _ in range(N)]

    plot_conc(y_pred)
