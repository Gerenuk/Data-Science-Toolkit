"""
This file contains the overview plotting function used in this project.
It is not meant to be re-used, but finds some usage for notebooks for visualization.

`plotfeat2` is mainly used. It plots a scatter matrix, the time evolution of all features in the given DateFrame
on selected days and optionally on some other random example days.
"""

import time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import datetime as dt
import math
from cytoolz import partition_all


def plotfeat(
    series, dates=None, num_plots=3, fig_width=15, fig_height=8, num_background_days=20
):
    """
    Used by `plotfeat2` below
    """
    if dates is None:
        dates=15
    
    random.seed(123)
    all_dates = set(series.index.date)

    if isinstance(dates, int):
        dates = [dt.datetime.combine(date, dt.datetime.min.time()) for date in sorted(random.sample(all_dates, min(dates, len(all_dates))))]
    else:
        dates = [dt.datetime.combine(date, dt.datetime.min.time()) if isinstance(date, dt.date) else date for date in dates]
        dates = sorted(dates)

    if len(dates) < num_plots:
        num_plots = len(dates)

    num_per_plot = int(math.ceil(len(dates) / num_plots))

    partitioned_dates = list(partition_all(num_per_plot, dates))

    feat_name = series.name

    fig, axs = plt.subplots(
        ncols=num_plots, figsize=(fig_width * num_plots, fig_height), squeeze=False
    )

    ys = []
    midnight = dt.time(0, 0)

    for ax, plot_dates in zip(axs.ravel(), partitioned_dates):
        style_cycler =  mpl.rcParams["axes.prop_cycle"] #cycler.cycler("color", plt.cm.Set2.colors)

        for date, style in zip(plot_dates, style_cycler):
            date_str = date.strftime("%Y-%m-%d")

            dd = series.loc[date_str]

            ax.plot(dd.index.time, dd, label=date_str, **style)

            for mark_date in dates:
                if mark_date.date() == date.date():
                    time = mark_date.time()
                    if time != midnight:
                        ax.axvline(time, ls=":", **style)

            ys.append(dd)

        if num_background_days is not None:
            num_background_days = min(num_background_days, len(all_dates))
            for date in random.sample(all_dates, num_background_days):
                date_str = date.strftime("%Y-%m-%d")
                dd = series.loc[date_str]

                ax.plot(dd.index.time, dd, c="gray", alpha=0.1, label="")

        ax.set_ylim(min(y.min() for y in ys), max(y.max() for y in ys))

        ax.set_title(feat_name)

        ax.legend()

    return axs


def plotfeat2(dataframe, highlight_times=None, style_cycler=None, max_plot_samples=1000, plot_time_dep=True, plot_extra_days=False, plot_scatter_matrix=True):  
    """
    dataframe: DataFrame from which all columns will be plotted on selected days
    highlight_times: Time to highlight in the plots. Can be anything that `pd.Timestamp` accepts. If the is a time component (not just a date), a mark will shown. For dates just these dates are selected
    plot_time_dep: Whether to plot the time dependences
    plot_scatter_matrix: Whether to plot the scatter matrix of all vs all features
    plot_extra_days: Whether to plot some extra sample days to show behaviour other than highlight_times
    max_plot_samples: How many samples to plot in the scatter matrix as a background
    """
    if isinstance(dataframe, pd.Series):
        dataframe=dataframe.to_frame()
    
    num_cols = len(dataframe.columns)
    
    if highlight_times is not None:
        highlight_times = [time if not isinstance(time, str) else pd.Timestamp(time) for time in highlight_times]
    
    if style_cycler is None:
        style_cycler = mpl.rcParams["axes.prop_cycle"]  # cycler.cycler("color", plt.cm.Set2.colors)  
    
    if plot_scatter_matrix and num_cols > 1:
        if len(dataframe)>max_plot_samples:
            plot_dataframe=dataframe.sample(max_plot_samples)
        else:
            plot_dataframe=dataframe
            
        snsgrid = plot_scatter_matrix2(plot_dataframe, all_kwargs={"color": "lightgray"})
        
        midnight=dt.time(0, 0)
        
        if highlight_times is not None:
            date_time_dict=cgroupby(lambda x:x.date(), highlight_times)
            
            for (date, times), style in zip(date_time_dict.items(), style_cycler):
                dataframe_day = dataframe[date.strftime("%Y-%m-%d")]

                if num_cols > 2:
                    plot_func=snsgrid.map_offdiag
                else:
                    plot_func=snsgrid.plot_joint

                plot_func(plot_other_data_on_snsgrid(plt.scatter, dataframe_day), **style)
                
                for time in times:
                    daytime=time.time()
                    
                    if daytime != midnight:
                        dataframe_daytime=dataframe.loc[time].to_frame().T
                        plot_func(plot_other_data_on_snsgrid(plt.scatter, dataframe_daytime), color="c")
                    
    plt.show()
       
    if plot_time_dep:
        for col in dataframe.columns:
            plotfeat(dataframe[col], dates=highlight_times)
            plt.suptitle("Selected days")
            plt.show()
            
            if plot_extra_days:
                plotfeat(dataframe[col])
                plt.suptitle("Other days")
                plt.show()
