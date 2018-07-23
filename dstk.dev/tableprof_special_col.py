__author__ = 'asuchane'

import random
from random import randint
import datetime as dt
from itertools import groupby
from pprint import pprint
import unicodedata
from collections import Counter

#data = [dt.datetime(randint(2000, 2015), randint(1, 12), randint(1, 28), randint(0, 23), randint(0, 59),
#                    randint(0, 59)).strftime("%Y-%b-%M" if random.random() < 0.5 else "%H:%M:%S"
#                                             ) for _ in range(1000)] + ["Hallo Test", "Hallo Keintest"]
#pprint(data[:20])


def pattern_parser(data, char_type_mapper, category_value_combiner_factory):
    pattern_result_map = {}
    pattern_counter = Counter()

    for row in data:
        char_groups = [(chartype, list(chargroup)) for chartype, chargroup in groupby(row, key=char_type_mapper)]
        char_signature = tuple(chartype for chartype, _ in char_groups)
        pattern_counter[char_signature] += 1

        if char_signature not in pattern_result_map:
            pattern_result_map[char_signature] = tuple(category_value_combiner_factory[chartype]() for chartype in
                                                       char_signature)

        cur_result = pattern_result_map[char_signature]
        for (_, char_group), cur_result_item in zip(char_groups, cur_result):
            cur_result_item.add(char_group)

    return pattern_counter, pattern_result_map


def result_formatter(pattern_counter, pattern_result_map, num_top_freq=2, join_sep=" / "):
    most_freq_items = pattern_counter.most_common(num_top_freq)
    result_lines = []

    for signature, num_pattern in most_freq_items:
        line = join_sep.join(combiner.result() for combiner in pattern_result_map[signature])
        result_lines.append("{}x {}".format(num_pattern, line))

    if len(pattern_counter) > num_top_freq:
        result_lines.append("...")

    return "\n".join(result_lines)


def chartype_mapper(char):
    if char == " ":
        return "space"
    elif char in "0123456789":
        return "digit"
    else:
        unicode_category = unicodedata.category(char)
        if unicode_category[0] == "L":
            return "letter"
    return "other"


class DistinctCombiner:
    def __init__(self, name=None):
        self.name = name
        self.values = Counter()

    def add(self, value):
        value = "".join(value)
        self.values[value] += 1

    def result(self):
        if len(self.values) == 1:
            return next(iter(self.values.items()))[0]
        return "[{}*{}]".format(self.values.most_common(1)[0][0]+".." if self.name is None else self.name, len(self.values))


class DigitCombiner:
    def __init__(self):
        self.min_number = float("inf")
        self.max_number = float("-inf")

    def add(self, value):
        value = int("".join(value))
        self.min_number = min(self.min_number, value)
        self.max_number = max(self.max_number, value)

    def result(self):
        return "[{}..{}]".format(self.min_number, self.max_number)


category_value_combiner = {
    "space": lambda: DistinctCombiner(" "),
    "digit": lambda: DigitCombiner(),
    "letter": lambda: DistinctCombiner(),
    "other": lambda: DistinctCombiner("?"),
}

if __name__ == '__main__':
    import csv, glob

    filenames=list(glob.glob(r"Q:\SAS\advanced_analytics\prod\Projects\International Business Partners\Data collection\International Transactions (FCP)\*.csv"))
    for filename in filenames:
        print("\n+++ FILE {} +++".format(filename))
        with open(filename) as f:
            reader=csv.DictReader(f)
            rows=list(reader)
            colnames=rows[0].keys()
            cols={colname:[row[colname] for row in rows] for colname in colnames}
            for colname, coldata in cols.items():
                result = pattern_parser(coldata, chartype_mapper, category_value_combiner)
                result_text = result_formatter(*result, join_sep="", num_top_freq=10)
                print("{} ---->".format(colname))
                print(result_text)
