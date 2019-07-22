"""
Recommended:
Set1_9
(Set2_8)
(Paired_12)
(Tableau_20)  # paired

Palettable: https://jiffyclub.github.io/palettable/
Matplotlib: http://matplotlib.org/examples/color/colormaps_reference.html
"""


# Kelly ['#f2f3f4', '#222222', '#f3c300', '#875692', '#f38400', '#a1caf1', '#be0032', '#c2b280', '#848482', '#008856',
# '#e68fac', '#0067a5', '#f99379', '#604e97', '#f6a600', '#b3446c', '#dcd300', '#882d17', '#8db600', '#654522', '#e25822', '#2b3d26']
# colorobject:


class ColorsDepleted(Exception):
    pass


def _get_named_colormap(colors):
    import matplotlib as mpl
    return mpl.colors.get_cmap(colors).colors
    # try:
    #     import palettable
    #
    #     module = palettable.colorbrewer.qualitative
    #     available_names = [(palettable, name) for name in dir(palettable) if
    #                        not name.startswith("__") and name not in ("palette", "version")]
    #     for name in dir(module):
    #         if not name.startswith("_"):
    #             available_names.append((palettable.colorbrewer.qualitative, name))
    # except ImportError:
    #     pass

    # try:
    # import colorobject
    #
    #     module = colorobject.custom_colorsets
    #     for name in dir(module):
    #         if not name.startswith("_"):
    #             return getattr(module, colors)
    # except ImportError:
    #     pass

    raise ValueError("Colormap {} not found")


class BaseColorMapper:
    def __init__(self, colors, fixed_map=None):
        """
        :param colors: finite list of available colors
        :param fixed_map:
        :return:
        """
        if isinstance(colors, str):
            colors = _get_named_colormap(colors)

        if fixed_map is not None:
            self.colormap = dict(fixed_map)  # keeps track of all color mappings
            self.all_free_colors = [c for c in colors if
                                    c not in fixed_map.values()]  # make all colors not in fixed_map available to choose from
        else:
            self.colormap = {}
            self.all_free_colors = list(colors)
        self.used_color_indices = set()

    def __call__(self, val):
        if val in self.colormap:
            return self.colormap[val]

        # get and set new color
        try:
            new_color_idx = self._get_free_color_idx(val)
        except ColorsDepleted:
            self.used_color_indices = set()
            new_color_idx = self._get_free_color_idx(val)

        self.used_color_indices.add(new_color_idx)
        new_color = self.all_free_colors[new_color_idx]
        self.colormap[val] = new_color
        return new_color

    def map(self, seq):
        return [self[s] for s in seq]


class ColorMapper(BaseColorMapper):
    def _get_free_color_idx(self, val):
        for i in range(len(self.all_free_colors)):
            if i not in self.used_color_indices:
                return i
        raise ColorsDepleted()


class HashColorMapper(BaseColorMapper):
    def _get_free_color_idx(self, val):
        num_colors = len(self.all_free_colors)
        start_i = hash(val) % num_colors
        for di in range(num_colors):
            i = (start_i + di) % num_colors
            if i not in self.used_color_indices:
                return i
        raise ColorsDepleted()


if __name__ == '__main__':
    c = HashColorMapper([1, 2, 3, 4], {"a": 2})
    for x in "abcdef":
        print(x, c[x])