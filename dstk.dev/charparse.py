import itertools as itoo
from collections import defaultdict, Counter
from operator import itemgetter
import re
import unicodedata

import pandas as pd


"""
TODO:
* unicode
* check if other rules match
"""


def charparserows(rows, header):
    for head, col in zip(header, zip(*rows)):
        print("*** {} ***".format(head))
        charparse(col)


def charparse(data):
    if isinstance(data, pd.DataFrame):
        for col in data:
            print("*** {} ***".format(col))
            charparse(data[col])
        return
    s = StringClasses(UnicodeCategoryClass())
    s.parse(data)
    s.report()
    return s


class UnicodeCategoryClass:
    def __init__(
        self,
        translate={
            "L": "Alph",
            "C": "Ctrl",
            "M": "Mark",
            "N": "Num",
            "P": "Punct",
            "S": "Sym",
            "Z": "Space",
        },
        subcat=False,
    ):
        self.subcat = subcat
        self.translate = translate

    def test(self, text):
        cat = unicodedata.category(text)
        if not self.subcat:
            cat = cat[0]
        if self.translate is not None:
            cat = self.translate.get(cat, cat)
        return cat

    def __str__(self):
        return "Unicode()"


class RegexCharClass:
    def __init__(self, name, regex):
        self.name = name
        self.regex = regex

    def test(self, text):
        if re.match(self.regex, text):
            return self.name
        else:
            return None

    def __str__(self):
        return "{{}}=regex({})".format(self.name, self.regex)


class CharClass:
    def __init__(self, name, chars):
        self.name = name
        self.chars = set(chars)

    def test(self, text):
        if text in self.chars:
            return self.name
        else:
            return None

    def __str__(self):
        return "{}=set({})".format(self.name, "".join(sorted(self.chars)))


def count_summary(counts):
    if len(counts) == 0:
        return ""
    diff_counts = set(counts)
    if len(diff_counts) == 1:
        return "{}".format(diff_counts.pop())
    sorted_counts = sorted(counts)
    if len(sorted_counts) == 2:
        return "{},{}".format(*sorted_counts)
    elif len(sorted_counts) == 3:
        return "{},{},{}".format(*sorted_counts)
    minc = sorted_counts[:2]
    maxc = sorted_counts[-2:]
    min_text = str(minc[0]) if minc[0] == minc[1] else "{},{}".format(*minc)
    max_text = str(maxc[1]) if maxc[0] == maxc[1] else "{},{}".format(*maxc)
    return min_text + "-" + max_text


class StringClasses:
    def __init__(self, *classes, count_summary=count_summary):
        self.classes = classes
        self.tag_count_D = defaultdict(list)
        self.count_summary = count_summary
        self.empty = 0
        self.all_class_counts = Counter()
        self.all_char_counts = Counter()
        self.line_class_counts = Counter()
        self.line_char_counts = Counter()
        self.unknown_char = []
        self.line_lengths = []
        self.num_lines = 0

    def parse(self, texts):
        for text in texts:
            self.num_lines += 1
            if text == "":
                self.empty += 1
                continue
            self.line_lengths.append(len(text))
            tags = []
            for char in text:
                self.all_char_counts[char] += 1
                for testclass in self.classes:
                    name = testclass.test(char)
                    if name is not None:
                        self.all_class_counts[name] += 1
                        tags.append(name)
                        break
                else:
                    tags.append("?")
                    self.all_class_counts["?"] += 1
                    self.unknown_char.append(char)
            tag_base = []
            counts = []
            matches = []
            for k, group in itoo.groupby(zip(tags, text), key=itemgetter(0)):
                tag_base.append(k)
                textmatch = "".join(c for _, c in group)
                counts.append(len(textmatch))
                matches.append(textmatch)
            tag_base = tuple(tag_base)
            for tag in set(tag_base):
                self.line_class_counts[tag] += 1
            for char in set(text):
                self.line_char_counts[char] += 1
            self.tag_count_D[tag_base].append((counts, text, matches))

    def get_class_seq(self, tags):
        return list(map(itemgetter(1), self.tag_count_D[tuple(tags)]))

    def get_class(self, name):
        return list(
            itoo.chain.from_iterable(
                map(itemgetter(1), data)
                for tags, data in self.tag_count_D.items()
                if name in tags
            )
        )

    def report(self):
        print("CHARREPORT", " ".join(str(c) for c in self.classes))
        sorted_line_lengths = sorted(self.line_lengths)
        min_lengths = sorted_line_lengths[:2]
        max_lengths = sorted_line_lengths[-2:]
        print(
            "{} lines; length {}".format(
                self.num_lines, count_summary(self.line_lengths)
            ),
            "; {} empty".format(self.empty) if self.empty > 0 else "",
            sep="",
        )
        tag_counts = sorted(
            ((len(data), tag) for tag, data in self.tag_count_D.items()), reverse=True
        )

        def value_summary(values, tag):
            if tag == "Num":
                values = list(map(float, values))
                minval = min(values)
                maxval = max(values)
                if minval != maxval:
                    return "({}-{})".format(minval, maxval)
                else:
                    return "({})".format(minval)
            elif tag in ["Alph", "Mark", "Punct", "Sym"]:
                charcounts = Counter(itoo.chain.from_iterable(values))
                return "({})".format(
                    "".join(map(itemgetter(0), charcounts.most_common(3)))
                    + (".." if len(charcounts) > 3 else "")
                )
            return ""

        for length, tag in tag_counts:
            counts = map(itemgetter(0), self.tag_count_D[tag])
            values = list(zip(*map(itemgetter(2), self.tag_count_D[tag])))
            counts_per_class = list(zip(*counts))

            print(
                "{}x {}".format(
                    length,
                    " ".join(
                        "{}{}:{}".format(
                            tag, value_summary(vals, tag), self.count_summary(counts)
                        )
                        for tag, counts, vals in zip(tag, counts_per_class, values)
                    ),
                )
            )
        print(
            "TotalClassLines",
            " ".join(
                "{}:{}({})".format(tag, count, self.all_class_counts[tag])
                for tag, count in self.line_class_counts.most_common()
            ),
        )
        print(
            "TotalCharLines",
            " ".join(
                "{}:{}({})".format(char, count, self.all_char_counts[char])
                for char, count in self.line_char_counts.most_common(10)
            ),
        )
        print("AllChars '{}'".format("".join(sorted(set(self.all_char_counts)))))
        print(
            "MostChars '{}'".format(
                "".join(map(itemgetter(0), self.all_char_counts.most_common()))
            )
        )
        if len(self.unknown_char) > 0:
            print("ClasslessChars '{}'".format("".join(sorted(set(self.unknown_char)))))


if __name__ == "__main__":
    # s=StringClasses(UnicodeCategoryClass()) #, CharClass("A", list("ab")), CharClass("B", list("12")))
    # s.parse(["aab1", "z12", "y12", "ab12a", "zzzzz", "aaaaaa", "a1-1", "b23--3", "z", "", "", "b"])
    # s.report()
    # print(s.get_class_seq("A B A".split()))
    # print(s.get_class("?"))
    import csv

    filename = r"F:\V\VT_Oberursel\RBV\KA\CRM_DM_Mafo\04_CMI\02_Aufgaben\04_Datenquellen\12_SAP_Reiseaufträge, Kundendaten\RA_Abzug_ab20110101_ALL_lines10000.csv"
    reader = csv.reader(open(filename, newline=""), delimiter=";")
    header = next(reader)
    charparserows(reader, header)
