import logging
import os
import pickle
import itertools as it
import scipy
import pandas as pd

logg = logging.getLogger(__name__)


class FuncCache:
    """
    @FuncCache()
    def func(..):
        ...
    result=func(..)
    """

    def __init__(self, dirname="", filename=None):
        self.dirname = dirname
        self.filename = filename

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            funcname = func.__name__
            if self.filename is not None:
                filename = self.filename
            else:
                filename = "{}.pkl".format(funcname)

            fullfilename = os.path.join(self.dirname, filename)

            if os.path.exists(fullfilename):
                logg.info(
                    "Loading function {} from cache file {} (function call skipped)".format(
                        funcname, fullfilename
                    )
                )
                result = pickle.load(open(fullfilename, "rb"))
                return result
            else:
                result = func(*args, **kwargs)
                logg.info(
                    "Storing result of function {} to cache file {}".format(
                        funcname, fullfilename
                    )
                )
                pickle.dump(result, open(fullfilename, "wb"))
                return result

        return wrapped


def cache_pkl(func):  # not used anymore?
    def wrapped(*args, **kwargs):
        func_name = func.__name__
        file_name = "{}.pkl".format(func_name)

        if os.path.exists(file_name):
            file = open(file_name, "rb")
            cache_args = pickle.load(file)
            cache_kwargs = pickle.load(file)
            if args == cache_args and kwargs == cache_kwargs:
                logg.info(
                    "Reading result of function {} from file {} since matching parameters found".format(
                        func_name, file_name
                    )
                )
                result = pickle.load(file)
                file.close()
                return result
            else:
                logg.info(
                    "Re-executing function {} since parameters have changed".format(
                        func_name
                    )
                )
                result = func(*args, **kwargs)
            file.close()
        else:
            logg.info(
                "Executing function {} since no cache file found".format(func_name)
            )
            result = func(*args, **kwargs)

        with open(file_name, "wb") as file:
            pickle.dump(args, file)
            pickle.dump(kwargs, file)
            pickle.dump(result, file)
            logg.info(
                "Stored result of function {} to file {}".format(func_name, file_name)
            )
        return result

    return wrapped


def thin_out(*datas, **kwargs):
    """
    Output at most #max_len tuples from zip(*datas) such that first element is
    evenly separated in values=datas[0]. Always includes first and last element. May output
    less than #max_len elements if datas shorter or very skewed in vals (e.g. 1,1,1,5).

    If val_max is not provided, it will iterate over datas to find it (possibly exhausting any generator!)

    datas[0] should be *sorted* and will be used for evenly spacing
    """
    max_len = kwargs.get("max_len", None)
    val_max = kwargs["val_max"] if "val_max" in kwargs else max(datas[0])

    data_iter = zip(*datas)

    val = next(data_iter)
    yield val

    val_min = val[0]

    estimates = [i * (val_max - val_min) / (max_len - 1) for i in range(1, max_len)]
    # not adding data_min yet, to avoid floating point issues with val-datamin+datamin
    # also keep this multiplication order to work out with the all-integer corner-case

    for estimate in estimates:
        if estimate <= val[0] - val_min:  # do not keep duplicate x values
            continue

        while val[0] - val_min < estimate:
            val = next(data_iter)  # _new to do check on ordering, too

        if val[0] == val_max:
            for val in data_iter:  # soak up rest if val_max reached
                pass

            yield val  # output last element
            break

        yield val


def dict_val_product(d, exclude_keys=[]):
    """
    Every list value in d.values() will be expanded to all cross-product combinations
    A list of dicts is returned

    >>> dict_val_product({"a":[1,2], "b":3, "c":[3,4]})
    [ { a : 1,  b : 3,  c : 3 }
      { a : 1,  b : 3,  c : 4 }
      { a : 2,  b : 3,  c : 3 }
      { a : 2,  b : 3,  c : 4 } ]
    """
    product_keys = [
        k for k, v in d.items() if k not in exclude_keys and isinstance(v, list)
    ]
    product_values = [d[k] for k in product_keys]
    non_product_dict = {k: v for k, v in d.items() if k not in product_keys}

    result = []

    for product_value_tuple in it.product(*product_values):
        product_key_dict = {k: v for k, v in zip(product_keys, product_value_tuple)}
        product_key_dict.update(non_product_dict)
        result.append(product_key_dict)

    return result


def compress_int_seq(vals, sep="-"):
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
            result.extend(map(str, group))
        else:
            result.append(f"{group[0]}{sep}{group[-1]}")

    return result


def fit_linear_models(dataframe):
    """
    Fit linear models to all pairs of columns.
    Useful for detecting Y=a*X+b relations by looking at the r_value
    """
    rows = []
    for col_name1, col_name2 in tqdm(list(combinations(dataframe.columns, r=2))):
        fit_result = scipy.stats.linregress(dataframe[col_name1], dataframe[col_name2])
        rows.append((col_name1, col_name2) + fit_result)

    result = pd.DataFrame(
        rows,
        columns=["col1", "col2", "slope", "intercept", "r_value", "p_value", "std_err"],
    ).sort_values("r_value", ascending=False)

    return result


def relation(df, col1, col2):
    result = (
        df.groupby([col1, col2])
        .size()
        .groupby(col1)
        .agg(["sum", "size", "max", "min"])
        .rename(
            columns={
                "sum": "n_inst",
                "size": "nunique_col2",
                "max": "n_inst_largest",
                "min": "n_inst_smallest",
            }
        )
    )
    result["n_inst_largest_frac"] = result["n_inst_largest"] / result["n_inst"]
    result["n_inst_smallest_frac"] = result["n_inst_smallest"] / result["n_inst"]
    return result

