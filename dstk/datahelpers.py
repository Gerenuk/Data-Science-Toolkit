from dataclasses import dataclass
from typing import Any
import numpy as np
from itertools import groupby
import re
from operator import itemgetter


@dataclass
class Bins:
    bin_min: float
    bin_max: float
    bin_num: int
    width: float = None
    edges: Any = None
    torch_kwargs: dict = None

    def __post_init__(self):
        self.width = (self.bin_max - self.bin_min) / self.bin_num
        self.edges = np.linspace(self.bin_min, self.bin_max, self.bin_num)
        self.torch_kwargs = {
            "bins": self.bin_num,
            "min": self.bin_min,
            "max": self.bin_max,
        }

    def __getitem__(self, idx):
        return self.bin_min + idx * self.width


def compress_alphanum_seq(vals, sep="-"):
    vals = sorted(vals, key=natural_sort)

    alpha_nums = []
    for v in vals:
        m = re.match("([^0-9]*)([0-9]+)$", v)
        if m:
            alpha_nums.append((m.group(1), m.group(2)))
        else:
            alpha_nums.append((v, None))

    result = []
    for alpha, a_alpha_nums in groupby(alpha_nums, key=itemgetter(0)):
        a_alpha_nums = [num for _, num in a_alpha_nums]
        if a_alpha_nums != [None]:
            for num_tuple in compress_int_seq(map(int, a_alpha_nums)):
                result.append(f"{alpha}{sep.join(map(str,num_tuple))}")
        else:
            result.append(alpha)

    return result


def compress_int_seq(vals):
    # TODO: show only last digits for run end?
    groups = []
    vals = sorted(vals)
    group = [vals[0]]

    for val in vals[1:]:
        if val == group[-1] + 1:
            group.append(val)
        else:
            groups.append(group)
            group = [val]
    groups.append(group)

    result = []
    for group in groups:
        if len(group) <= 2:
            result.extend((x,) for x in group)
        else:
            result.append((group[0], group[-1]))

    return result


def natural_sort(val):
    """
    sort v2 < v12
    """
    return tuple(
        int("".join(gr)) if is_digit else "".join(gr)
        for is_digit, gr in groupby(val, lambda x: x.isdigit())
    )


def factorize(dd):
    pass