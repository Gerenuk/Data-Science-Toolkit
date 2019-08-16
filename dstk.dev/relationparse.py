import itertools as itoo
from collections import defaultdict


def relationparse(cols):
    for (cname1, col1), (cname2, col2) in itoo.combinations(cols, 2):
        try:
            rel12 = relationparse_pair(col1, col2)
            rel21 = relationparse_pair(col2, col1)
            if rel12 and rel21:
                print("{} oneToOne {}".format(cname1, cname2))
            elif rel12:
                print("{} superclassOf {}".format(cname1, cname2))
            elif rel21:
                print("{} superclassOf {}".format(cname2, cname1))
        except Exception as e:
            print("Failed parsing {}, {} due to {}".format(cname1, cname2, e))


def relationparse_pair(x, y):
    """
    return True if x is superclass of y; or each y value has only one x values assign
    """
    collect = defaultdict(set)
    for x_e, y_e in zip(x, y):
        collect[y_e].add(x_e)
        if len(collect[y_e]) > 1:
            return False
    return True


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame([[1, 2, 3], [1, 2, 3], [2, 4, 5], [2, 4, 5]], columns=list("abc"))
    relationparse(df.iteritems())
