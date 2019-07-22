import logging

import matplotlib as mpl

try:
    import palettable

    for cmap_type, modL in [("seq", (palettable.colorbrewer.sequential, palettable.colorbrewer.diverging)),
                            ("qual", (palettable.colorbrewer.qualitative, palettable.tableau, palettable.wesanderson))]:
        for mod in modL:
            for name in dir(mod):
                if name.startswith("_") or name in ["get_map", "print_maps", "wesanderson", "absolute_import",
                                                    "tableau"]:
                    continue
                try:
                    if cmap_type == "seq":
                        cmap = getattr(mod, name).mpl_colormap
                    elif cmap_type == "qual":
                        cmap = mpl.colors.ListedColormap(getattr(mod, name).mpl_colors)
                    else:
                        raise ValueError()
                    mpl.cm.register_cmap(name, cmap)
                except AttributeError:
                    logging.warning("Failed to import colormap '{}' from {}".format(name, mod.__name__))
except ImportError as e:
    logging.warning("Did not import colormaps from palettable due to import error: {}".format(e))

try:
    import colorobject
    for cmap_type, modL in [("seq", (colorobject.custom_colormaps,)),
                            ("qual", (colorobject.custom_colorsets,))]:
        for mod in modL:
            for name in dir(mod):
                try:
                    if name.startswith("_") or name in ["index", "count"]:
                        continue
                    if cmap_type == "seq":
                        cmap = getattr(mod, name)
                    elif cmap_type == "qual":
                        cmap = mpl.colors.ListedColormap([(r, g, b) for r, g, b, a in getattr(mod, name)])
                    mpl.cm.register_cmap(name, cmap)
                except AttributeError:
                    logging.warning("Failed to import colormap '{}' from {}".format(name, mod.__name__))
except ImportError as e:
    logging.warning("Did not import colormaps from colorobject due to import error: {}".format(e))


kelly_colors = ["#FFB300",  # Vivid Yellow
                "#803E75",  # Strong Purple
                "#FF6800",  # Vivid Orange
                "#A6BDD7",  # Very Light Blue
                "#C10020",  # Vivid Red
                "#CEA262",  # Grayish Yellow
                "#817066",  # Medium Gray

                # The following don't work well for people with defective color vision
                "#007D34",  # Vivid Green
                "#F6768E",  # Strong Purplish Pink
                "#00538A",  # Strong Blue
                "#FF7A5C",  # Strong Yellowish Pink
                "#53377A",  # Strong Violet
                "#FF8E00",  # Vivid Orange Yellow
                "#B32851",  # Strong Purplish Red
                "#F4C800",  # Vivid Greenish Yellow
                "#7F180D",  # Strong Reddish Brown
                "#93AA00",  # Vivid Yellowish Green
                "#593315",  # Deep Yellowish Brown
                "#F13A13",  # Vivid Reddish Orange
                "#232C16",  # Dark Olive Green
]
# kelly_colors=['#f2f3f4', '#222222', '#f3c300', '#875692', '#f38400', '#a1caf1', '#be0032', '#c2b280', '#848482', '#008856',
# '#e68fac', '#0067a5', '#f99379', '#604e97', '#f6a600', '#b3446c', '#dcd300', '#882d17', '#8db600', '#654522', '#e25822', '#2b3d26']
# these are numbers from the paper; wrong conversion?
mpl.cm.register_cmap("kelly", mpl.colors.ListedColormap(kelly_colors))  # check if remove black/white at beginning


def plot_qualitative_colormaps(cmap_list):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(nrows=len(cmap_list))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)

    for ax, name in zip(axes, cmap_list):
        cmap=mpl.cm.get_cmap(name)
        ax.pcolor(np.array([list(range(cmap.N))]), cmap=cmap)
        ax.set_frame_on(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for i in range(cmap.N):
            ax.text(0.5+i,0.5,i, horizontalalignment='center', verticalalignment='center')
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


def plot_sequential_colormaps(cmap_list):
    import matplotlib.pyplot as plt
    import numpy as np
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, axes = plt.subplots(nrows=len(cmap_list))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)

    for ax, name in zip(axes, cmap_list):
        try:
            cmap=plt.get_cmap(name)
        except (ValueError, AttributeError):
            print("Fail")
            try:
                cmap=getattr(palettable.colorbrewer.sequential,name).mpl_colormap
            except (ValueError, AttributeError):
                cmap=getattr(palettable.colorbrewer.diverging,name).mpl_colormap
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plot_qualitative_colormaps(["kelly", "Set1_9",  "Set2_8", "Paired_12", "Tableau_20"])
    plt.show()

    plot_sequential_colormaps(
        ["afmhot_r",
         "YlOrRd",
         "YlGn",
         "bone_r",
         "copper_r",
         "autumn_r",
         "cubehelix_r",
         "BuPu_9",
         "OrRd_9",
         "RdPu_9",
         "YlGnBu_9",
         "YlOrRd_9",
         "Spectral",
         "RdYlGn",
         "BrBG",
         "cool2hot",
         "deep2shallow",
         "sparse2dense",
         "low2high",
         ])
    plt.show()

