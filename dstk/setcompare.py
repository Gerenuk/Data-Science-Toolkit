from collections import Counter
from cytoolz import pluck
from ipy_table import make_table, apply_theme, set_cell_style
from itertools import product

count_style = ' style="text-align:right"'
# count_style=""  # use this if you haven't hacked ipy_tables and remove &nbsp substitution


def format_tuples(keys, *cntrs):
    result = ["<table>"]
    for key in keys:
        result.append(
            "<tr> <td>{}</td> {} </tr>".format(
                key,
                " ".join(
                    "<td{}> &#215 {}</td>".format(count_style, cntr[key])
                    for cntr in cntrs
                ),
            )
        )
    result.append("</table>")
    return "".join(result)


def setcompare(iter1, iter2):
    cntr1 = Counter(iter1)
    cntr2 = Counter(iter2)
    only1 = cntr1.keys() - cntr2.keys()
    only2 = cntr2.keys() - cntr1.keys()
    both = cntr1.keys() & cntr2.keys()

    cnt1 = sum(cntr1[key] for key in only1)
    cnt2 = sum(cntr2[key] for key in only2)
    cnt12a = sum(cntr1[key] for key in both)
    cnt12b = sum(cntr2[key] for key in both)
    distinct1 = len(only1)
    distinct2 = len(only2)
    distinct12 = len(both)

    cnt_perct = "{} ({:.0%})".format

    if hasattr(iter1, "name"):
        name1 = f"1 {iter1.name}"
    else:
        name1 = "1"

    if hasattr(iter2, "name"):
        name2 = f"2 {iter2.name}"
    else:
        name2 = "2"

    display_data = [
        ["", f"Set {name1} only", "Intersect.", f"Set {name2} only"],
        [
            "Count",
            cnt_perct(cnt1, cnt1 / (cnt1 + cnt12a)),
            "{} | {}".format(cnt12a, cnt12b),
            cnt_perct(cnt2, cnt2 / (cnt2 + cnt12b)),
        ],
        [
            "Distinct count",
            cnt_perct(distinct1, distinct1 / (distinct1 + distinct12)),
            distinct12,
            cnt_perct(distinct2, distinct2 / (distinct2 + distinct12)),
        ],
        [
            "Examples",
            format_tuples(
                pluck(0, Counter({key: cntr1[key] for key in only1}).most_common(5)),
                cntr1,
            ),
            format_tuples(
                pluck(
                    0,
                    Counter({key: cntr1[key] + cntr2[key] for key in both}).most_common(
                        5
                    ),
                ),
                cntr1,
                cntr2,
            ),
            format_tuples(
                pluck(0, Counter({key: cntr2[key] for key in only2}).most_common(5)),
                cntr2,
            ),
        ],
    ]

    make_table(display_data)
    table = apply_theme("basic_both")
    for x, y in product([0, 1, 2], [1, 2, 3]):
        set_cell_style(x, y, align="center")
    return table
