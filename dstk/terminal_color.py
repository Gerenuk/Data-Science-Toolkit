import colorful


colorful.use_true.colors()   # did not work in Jupyter on Windows?!


class Color:
    """
    Colors to use for terminal output
    e.g. `Color.red("TEXT")`
    """
    purple = colorful.purple
    red = colorful.tomato
    violet = colorful.violet
    blue = colorful.cornflowerBlue
    lightblue = colorful.deepSkyBlue
    green = colorful.limeGreen
    yellow = colorful.goldenrod1


def color_term_print():
    for color in [col for col in dir(Color) if not col.startswith("__")]:
        print(f"{getattr(Color, color)('*** TEXT TEXT TEXT *** '+color)}")


def print_rgb_colors(filename="/etc/X11/rgb.txt"):
    def process(text):
        text = "".join(word.capitalize() for word in text.split())
        text = text[0].lower() + text[1:]
        return text

    colors = sorted(
        set(
            process(line.split("\t\t")[-1][:-1])
            for line in open(filename).readlines()[1:]
        )
    )

    for col in sorted(set(colors)):
        if col.startswith(("grey", "gray")):
            continue
        try:
            print(getattr(colorful, col)("*** TEXT TEXT TEXT ***"), col)
        except:
            pass
