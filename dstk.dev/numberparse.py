from collections import Counter
import statistics

import numpy as np

from charparse import *


max_display_values = 20


def counts_report(data):
    counts = Counter(data)
    print("Distinct {}".format(len(counts)))
    most_common = counts.most_common()

    if len(most_common) <= max_display_values:
        print("Distinct frequent values {}".format(", ".join("{}".format(val) for val, _ in most_common)))
        print("Distinct ordered values {}".format(", ".join("{}".format(val) for val, _ in sorted(most_common))))

    if len(counts) <= 4:
        print("Counts {}".format(", ".join("{}({})".format(val, count) for val, count in most_common)))
    else:
        mostfreq = most_common[:2]
        leastfreq = most_common[-2:]
        print("Counts {}".format(
            "{}({}), {}({}) .. {}({}), {}({})".format(*itoo.chain.from_iterable(mostfreq + leastfreq))))


def size_report(data):
    ordered = sorted(set(data))

    counts = Counter(data)
    if len(ordered) <= 4:
        print("Counts {}".format(", ".join("{}({})".format(val, counts[val]) for val in ordered)))
    else:
        smallest = ordered[:2]
        largest = ordered[-2:]
        print("Ordered {}".format("{}({}), {}({}) .. {}({}), {}({})".format(
            *itoo.chain.from_iterable((val, counts[val]) for val in smallest + largest))))

    try:
        print("Quartiles {}".format(
            " - ".join(str(np.percentile(data, percent, interpolation="nearest")) for percent in [0, 25, 50, 75, 100])))
    except ValueError as e:
        print("No Quartiles due to {}".format(e))

    try:
        print("Average {:.2g}".format(statistics.mean(data)))
    except ValueError as e:
        print("No average due to {}".format(e))


class NumberParse:
    def __init__(self, parsefunc):
        self.parsefunc = parsefunc
        self.parsed_numbers = []
        self.unparsed = []

    def parse(self, data):
        for point in data:
            try:
                self.parsed_numbers.append(self.parsefunc(point))
            except ValueError:
                self.unparsed.append(point)

    def select_parsed_numbers(self, smallest=True, largest=True, mostcommon=True):
        exclude_numbers = []
        if not smallest:
            exclude_numbers.append(min(self.parsed_numbers))
        if not largest:
            exclude_numbers.append(max(self.parsed_numbers))
        if not mostcommon:
            exclude_numbers.append(Counter(self.parsed_numbers).most_common(1)[0][0])
        return [num for num in self.parsed_numbers if num not in exclude_numbers]

    def report(self):
        total_num = len(self.parsed_numbers) + len(self.unparsed)
        parsed_num = len(self.parsed_numbers)
        unparsed_num = len(self.unparsed)
        print("NUMBER REPORT", "{} values".format(total_num))
        print("Parsed {} ({:.1%})".format(parsed_num, parsed_num / total_num),
              "; Unparsed {} ({:.1%})".format(unparsed_num, unparsed_num / total_num) if unparsed_num else "", sep="")
        print("-- Parsed --")
        counts_report(self.parsed_numbers)
        size_report(self.parsed_numbers)
        print("-- Unparsed --")
        counts_report(self.unparsed)
        print("-- Charclass Unparsed --")
        sc = StringClasses(UnicodeCategoryClass())
        sc.parse(self.unparsed)
        sc.report()


if __name__ == '__main__':
    n = NumberParse(int)
    n.parse(["1", "b", "3", "b", "a", "z", "a", "c", "d", "4", "5", "6", "6"])
    n.report()
    print(n.select_parsed_numbers(smallest=False, mostcommon=False))