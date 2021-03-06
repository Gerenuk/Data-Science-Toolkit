from fractions import Fraction
import colorful
from statistics import mean, stdev
from math import sqrt, floor, log10


def cf_to_fraction(cf):
    result = Fraction(cf[-1], 1)
    for cf_val in reversed(cf[:-1]):
        result = 1 / result
        result += cf_val
    return result


def val_to_fractions(value, stop=1e-6, max_steps=20):
    """
    Could recover a/b with b up to at least 1000
    """
    assert value >= 0

    cf_vals = []
    result = []

    for _ in range(max_steps):
        cf_val = int(value)
        cf_vals.append(cf_val)

        result.append(cf_to_fraction(cf_vals))

        value -= cf_val
        if value < stop:
            break

        value = 1 / value

    return result


def uncertain_num_to_str(x, dx, use_color=True):
    """
    Error is always added at the end.
    """
    n = floor(log10(dx))
    dx = round(dx, -n)  # re-round in case dx 0.096 -> 0.1
    n = floor(log10(dx))

    x_str = str(round(x, -n))
    dx_str = "(" + str(round(dx * 10 ** (-n))) + ")"

    if use_color:
        result_str = x_str + colorful.gray(dx_str)
    else:
        result_str = x_str + dx_str

    return result_str


def mean_estimates_to_str(xs):
    return uncertain_num_to_str(mean(xs), stdev(xs) / sqrt(len(xs)))

