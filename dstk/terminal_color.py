import colorful


def color_term_print():
    colorful.use_true.colors()
    for color in ["purple",
                  "tomato",
                  "violet",
                  "cornflowerBlue",
                  "deepSkyBlue",
                  "limeGreen",
                  "goldenrod1"]:
        print(f"{getattr(colorful, color)('*** TEXT TEXT TEXT *** '+color)}")


def print_rgb_colors(filename="/etc/X11/rgb.txt"):
    def process(text):
        text = "".join(word.capitalize() for word in text.split())
        text = text[0].lower() + text[1:]
        return text

    colors = sorted(set(process(line.split("\t\t")[-1][:-1]) for line in open(filename).readlines()[1:]))

    colorful.use_true_colors()

    for col in sorted(set(colors)):
        if col.startswith(("grey", "gray")):
            continue
        try:
            print(getattr(colorful, col)("*** TEXT TEXT TEXT ***"), col)
        except:
            pass