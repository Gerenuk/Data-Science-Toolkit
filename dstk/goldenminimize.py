from collections import namedtuple
from math import sqrt
from typing import NamedTuple, Any


class GoldenInconsistent(Exception):
    def __init__(self, golden_state):
        self.golden_state = golden_state


class GoldenState(NamedTuple):
    a: Any
    b: Any
    c: Any
    new_x: Any
    ya: Any
    yb: Any
    yc: Any
    new_y: Any

    def __repr__(self):
        return f"GoldenState({self.a}, {self.b}, {self.c} -> {self.ya}, {self.yb}, {self.yc}; new: {self.new_x} -> {self.new_y}"


def golden_minimize(xs, ys=None, min_bound=True, max_bound=True):
    """
    Pass 2 or 3 initial values
    Range may extend within bounds if minimum appears at edge

    Example:
    def func(x):
        return x**2

    xs = [-10, 5]
    mini = golden_minimize(xs, [func(x) for x in xs])
    x, state = next(mini)
    for _ in range(10):
        x, state = mini.send(func(x))

    The case a b b a  with b < a is not solved at throws GoldenInconsistent
    """
    phi = (1 + sqrt(5)) / 2
    pos = 2 - phi  # ~0.382

    if ys is None:
        ys = []
        for x in xs:
            y = yield x, None
            ys.append(y)

    if len(xs) == 2:
        a, c = xs
        ya, yc = ys
        b = a + pos * (c - a)
        yb = yield b, GoldenState(a, b, c, None, ya, None, yc, None)
    else:
        a, b, c = xs
        ya, yb, yc = ys

    if min_bound is True:
        min_bound = a

    if max_bound is True:
        max_bound = c

    new_x = None
    new_y = None

    while 1:
        # print(">>>", a, b, c)
        assert a < b < c, "x coordinates not ordered"

        d1 = b - a
        d2 = c - b

        if ya < yb <= yc:  # extend region left
            if min_bound < a:
                new_x = b - d1 / pos

                if new_x < min_bound:
                    new_x = min_bound

                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc, new_y)
                a, b, c = new_x, a, b
                ya, yb, yc = new_y, ya, yb
            else:
                new_x = a + pos * (b - a)
                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc, new_y)
                b, c = new_x, b
                yb, yc = new_y, yb

        elif ya >= yb > yc:  # extend region right
            if c < max_bound:
                new_x = b + d2 / pos
                if new_x > max_bound:
                    new_x = max_bound

                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc, new_y)
                a, b, c = b, c, new_x
                ya, yb, yc = yb, yc, new_y
            else:
                new_x = c - pos * (c - b)
                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc, new_y)
                a, b = b, new_x
                ya, yb = yb, new_y

        elif ya >= yb and yb <= yc:
            if d1 < d2:
                new_x = c - (1 - pos) * (c - b)
                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc, new_y)

                if new_y > yc:
                    raise GoldenInconsistent(
                        GoldenState(a, b, c, new_x, ya, yb, yc, new_y)
                    )

                if new_y < yb:
                    a, b = b, new_x
                    ya, yb = yb, new_y
                elif new_y > yb:
                    c = new_x
                    yc = new_y
                else:
                    raise GoldenInconsistent(
                        GoldenState(a, b, c, new_x, ya, yb, yc, new_y)
                    )
            else:
                new_x = a + (1 - pos) * (b - a)
                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc, new_y)

                if new_y > ya:
                    raise GoldenInconsistent(
                        GoldenState(a, b, c, new_x, ya, yb, yc, new_y)
                    )

                if new_y < yb:
                    b, c = new_x, b
                    yb, yc = new_y, yb
                elif new_y > yb:
                    a = new_x
                    ya = new_y
                else:
                    raise GoldenInconsistent(
                        GoldenState(a, b, c, new_x, ya, yb, yc, new_y)
                    )
        else:
            raise GoldenInconsistent(GoldenState(a, b, c, new_x, ya, yb, yc, new_y))
