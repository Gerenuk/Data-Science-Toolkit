# -*- coding: utf-8 -*-

from operator import itemgetter
import itertools as itoo
import types

def subselect(df, var, crit, extra=None):
    return df[df[var].isin(getvals(df, var, crit, extra))]


class GetDfVals:
    def __init__(self, var, crit, extra=None, order=None):
        self.var=var
        self.crit=crit
        self.extra=extra or set()
        self.order=order

    def var(self):
        return self.var

    def val(self, df, order=None):
        group=df.groupby(self.var)
        g=self.crit(group)
        result=set(g[g].index)|set(self.extra)
        if order is None and self.order is not None:
            order=self.order
        if order is not None:
            result=list(order(group)[result].order(ascending=False).index)
        return result


class GetDfVallist:
    def __init__(self, var, val_list):
        self.var=var
        self.val_list=val_list

    def var(self):
        return self.var

    def val(self, df):
        return self.val_list


def split(df, *var_val_list, unpack_single=True):
    # TODO: split by function or (var,map) which creates key
    var_val_list_=[]
    for v in var_val_list:
        if isinstance(v, str):
            var_val_list_.append((v, set(df[v].unique())))
        else:
            var_val_list_.append((v.var, v.val(df)))

    var_list=list(map(itemgetter(0), var_val_list_))
    val_list_list=list(map(itemgetter(1), var_val_list_))
    for val_list in itoo.product(*val_list_list):
        #print(" & ".join("{}=={}".format(var, repr(val)) for var, val in zip(var_list, val_list)))
        yield (df.query(" & ".join("{}=={}".format(var, repr(val)) for var, val in zip(var_list, val_list))),
               val_list if not(unpack_single and len(val_list)==1) else val_list[0])
