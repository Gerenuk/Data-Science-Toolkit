import matplotlib as mpl


class ColorHash:
    """
    to be used with df.styler.applymap
    """
    def __init__(self, colors=mpl.cm.tab20.colors):
        n_colors=len(colors)
        self.color_spec={i:'background: rgb({})'.format(",".join(str(int(255*colval)) for colval in color)) for i, color in enumerate(colors)}
        
    def __call__(self, value):
        color_idx = hash(value) % n_colors
        return self.color_spec[color_idx]


    