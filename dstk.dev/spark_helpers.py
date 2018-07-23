from operator import add
from collections import Counter


def countspark_top(rdd, create_items, ascending=False, maxcollect=1000, val_func=lambda row: 1, agg_func=add):
    result = (rdd
              .flatMap(lambda row: [(match, val_func(row)) for match in create_items(row)])
              .reduceByKey(agg_func)
              .map(lambda t: (t[1], t[0]))
              .sortByKey(ascending=ascending))

    if maxcollect:
        result = Counter({val: cnt for cnt, val in result.take(maxcollect)})
    else:
        result = result.cache()

    return result


def countsparkdf_top(df, val_var, sum_var=None, ascending=False, maxcollect=1000):
    # offer unordered too
    df1 = df.groupby(val_var)
    df2 = df1.count() if sum_var is None else df1.sum(sum_var)
    df3 = df2.orderBy("count" if sum_var is None else "sum(" + sum_var + ")", ascending=ascending)

    if maxcollect:
        result = Counter({val: cnt for val, cnt in df3.take(maxcollect)})
    else:
        result = df3.cache()

    return result
