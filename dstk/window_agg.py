import numba
import numpy as np
from functools import reduce
from operator import add
import pandas as pd
from math import sqrt


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("cur_idx", numba.int64),
    ]
)
class ScalarResultStorage:
    def __init__(self, N):
        self.values = np.full(N, fill_value=np.nan, dtype=numba.float64)
        self.cur_idx = 0

    def add(self, value):
        self.values[self.cur_idx] = value
        self.cur_idx += 1


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("value", numba.float64),
    ]
)
class SumAgg:
    def __init__(self, values):
        self.values = values
        self.value = 0

    def add(self, idx):
        # print("Add", idx)
        self.value += self.values[idx]

    def remove(self, idx):
        # print("Remove", idx)
        self.value -= self.values[idx]

    def reset(self):
        # print("Reset")
        self.value = 0


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("cur_val", numba.float64),
        ("sum", numba.float64),
        ("sum_sq", numba.float64),
        ("n", numba.int16),
    ]
)
class ZScoreAgg:
    def __init__(self, values):
        self.values = values
        self.sum = 0
        self.sum_sq = 0
        self.n = 0
        self.cur_val = np.nan

    def add(self, idx):
        val = self.values[idx]
        self.cur_val = val
        self.sum += val
        self.sum_sq += val ** 2
        self.n += 1

    def remove(self, idx):
        val = self.values[idx]
        self.sum -= val
        self.sum_sq -= val ** 2
        self.n -= 1

    def reset(self):
        self.sum = 0
        self.sum_sq = 0
        self.n = 0

    @property
    def value(self):
        if self.n == 0:
            return np.nan
        elif self.n == 1:
            return self.sum
        else:
            mean = self.sum / self.n
            std = sqrt(self.sum_sq / self.n - mean ** 2)
            if std == 0:
                return 0  # should this be zero?
            else:
                return (self.cur_val - mean) / std


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("sum", numba.float64),
        ("n", numba.int16),
    ]
)
class MeanAgg:
    def __init__(self, values):
        self.values = values
        self.sum = 0
        self.n = 0

    def add(self, idx):
        val = self.values[idx]
        self.sum += val
        self.n += 1

    def remove(self, idx):
        val = self.values[idx]
        self.sum -= val
        self.n -= 1

    def reset(self):
        self.sum = 0
        self.n = 0

    @property
    def value(self, idx):
        if self.n == 0:
            return np.nan
        else:
            mean = self.sum / self.n
            return mean


@numba.jitclass([("value", numba.float64)])
class CountAgg:
    def __init__(self, N):
        self.value = 0

    def add(self, idx):
        self.value += 1

    def remove(self, idx):
        self.value -= 1

    def reset(self):
        self.value = 0


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("cnts", numba.types.DictType(numba.float64, numba.int64)),
    ]
)
class NuniqueAgg:
    def __init__(self, values):
        self.values = values
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )

    def add(self, idx):
        val = self.values[idx]

        if val in self.cnts:
            self.cnts[val] += 1
        else:
            self.cnts[val] = 1

    def remove(self, idx):
        val = self.values[idx]

        if self.cnts[val] == 1:
            del self.cnts[val]
        else:
            self.cnts[val] -= 1

    def reset(self):
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )

    @property
    def value(self, idx):
        return len(self.cnts)


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("cnts", numba.types.DictType(numba.float64, numba.int64)),
    ]
)
class MinAgg:
    def __init__(self, values):
        self.values = values
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )

    def add(self, idx):
        val = self.values[idx]

        if val in self.cnts:
            self.cnts[val] += 1
        else:
            self.cnts[val] = 1

    def remove(self, idx):
        val = self.values[idx]

        if self.cnts[val] == 1:
            del self.cnts[val]
        else:
            self.cnts[val] -= 1

    def reset(self):
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )

    @property
    def value(self, idx):
        return min(self.cnts)   # could be optimized


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("cnts", numba.types.DictType(numba.float64, numba.int64)),
    ]
)
class MaxAgg:
    def __init__(self, values):
        self.values = values
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )

    def add(self, idx):
        val = self.values[idx]

        if val in self.cnts:
            self.cnts[val] += 1
        else:
            self.cnts[val] = 1

    def remove(self, idx):
        val = self.values[idx]

        if self.cnts[val] == 1:
            del self.cnts[val]
        else:
            self.cnts[val] -= 1

    def reset(self):
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )

    @property
    def value(self, idx):
        return max(self.cnts)    # could be optimized


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("cur_val", numba.float64),
        ("cnts", numba.types.DictType(numba.float64, numba.int64)),
        ("num", numba.int64),
    ]
)
class FracAgg:
    def __init__(self, values):
        self.values = values
        self.cur_val = np.nan

        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )
        self.num = 0

    def add(self, idx):
        val = self.values[idx]
        self.cur_val = val

        self.num += 1

        if val in self.cnts:
            self.cnts[val] += 1
        else:
            self.cnts[val] = 1

    def remove(self, idx):
        val = self.values[idx]

        self.num -= 1

        if self.cnts[val] == 1:
            del self.cnts[val]
        else:
            self.cnts[val] -= 1

    def reset(self):
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )
        self.num = 0

    @property
    def value(self, idx):
        if self.num == 0:
            return np.nan
        else:
            return self.cnts[self.cur_val] / self.num


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("last_value", numba.float64),
        ("cur_value", numba.float64),
    ]
)
class DiffAgg:
    def __init__(self, values):
        self.values = values

        self.last_value = np.nan
        self.cur_value = np.nan

    def add(self, idx):
        val = self.values[idx]

        self.last_value = self.cur_value
        self.cur_value = val

    def reset(self):
        self.last_value = np.nan
        self.cur_value = np.nan

    @property
    def value(self, idx):
        return self.cur_value - self.last_value


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("value", numba.float64),
    ]
)
class LastNotNaAgg:
    def __init__(self, values):
        self.values = values

        self.value = np.nan

    def add(self, idx):
        val = self.values[idx]

        if not np.isnan(val):
            self.value = val

    def reset(self):
        self.value = np.nan


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("window_values", numba.float64[:]),
        ("insert_index", numba.int16),
        ("n", numba.int16),
        ("missing", numba.float64),
    ]
)
class DiffAggN:
    def __init__(self, values, n, missing=np.nan):
        self.values = values

        self.window_values = np.full(n, missing)
        self.insert_index = 0
        self.n = n
        self.missing = missing

    def add(self, idx):
        val = self.values[idx]

        self.window_values[self.insert_index] = val
        self.insert_index -= 1
        if self.insert_index < 0:
            self.insert_index = self.n - 1

    def reset(self):
        self.window_values = np.full(self.n, self.missing)
        self.insert_index = self.n - 1

    @property
    def value(self, idx):
        cur_index = self.insert_index + 1
        if cur_index >= self.n:
            cur_index = 0

        past_index = self.insert_index

        return (
            self.window_values[cur_index] - self.window_values[past_index]
        )


@numba.jitclass(
    [
        ("values", numba.float64[:]),
        ("value", numba.float64),
    ]
)
class MaxExpandAgg:
    def __init__(self, values):
        self.values = values

        self.value = np.nan

    def add(self, idx):
        val = self.values[idx]

        if np.isnan(self.value) or (not np.isnan(val) and val > self.value):
            self.value = val

    def reset(self):
        self.value = np.nan


@numba.jit(nogil=True, nopython=True)
def groupby_window_agg(group, time_vals, aggregator, timediff_start, timediff_end=0, store_flags=None):
    """
    start_idx and end_idx point at candidates for addition or removal
    aggregator will be controlled with .add, .remove, .store and is supposed to store the result
    needs to be sorted by [group, time_vals]
    """
    start_idx = 0
    end_idx = 0

    N = len(group)

    if store_flags is None:
        result_N = N
    else:
        result_N = np.sum(store_flags)

    result = ScalarResultStorage(result_N)

    cur_group = group[0]

    for cur_idx in range(N):
        # Track group variable
        new_group = group[cur_idx]

        if new_group != cur_group:
            aggregator.reset()

            start_idx = cur_idx
            end_idx = cur_idx

            cur_group = new_group

        cur_time = time_vals[cur_idx]

        # Adjust window end
        window_end_time = cur_time + timediff_end

        while (
            end_idx < N
            and group[end_idx] == cur_group
            and time_vals[end_idx] <= window_end_time
        ):
            aggregator.add(end_idx)
            end_idx += 1

        # Adjust window start
        window_start_time = cur_time + timediff_start

        while (
            start_idx < N
            and group[start_idx] == cur_group
            and time_vals[start_idx] < window_start_time
        ):
            aggregator.remove(start_idx)
            start_idx += 1

        if store_flags is None or store_flags[cur_idx] == 1:
            result.add(aggregator.value)

    return result.values


@numba.jit(nogil=True, nopython=True)
def groupby_expanding_agg(group, aggregator, store_flags=None):
    """
    start_idx and end_idx point at candidates for addition or removal
    aggregator will be controlled with .add, .remove, .store and is supposed to store the result
    needs to be sorted by [group]
    """
    N = len(group)

    cur_group = group[0]

    if store_flags is None:
        result_N = N
    else:
        result_N = np.sum(store_flags)

    result = ScalarResultStorage(result_N)

    for cur_idx in range(N):
        # Track group variable
        new_group = group[cur_idx]

        if new_group != cur_group:
            aggregator.reset()

            cur_group = new_group

        aggregator.add(cur_idx)

        if store_flags is None or store_flags[cur_idx] == 1:
            result.add(aggregator.value)

    return result.values


def cat_factorize(cols):
    if isinstance(cols, pd.DataFrame):
        cols = list(zip(*[cols[c] for c in cols]))

    cols = pd.factorize(cols)[0]

    return cols


def pd_groupby_window_agg(groups, time, agg, vals=None, start=np.nan, end=0, store_flags=None):
    groups = cat_factorize(groups)

    df_dict = {"group": groups, "time": time}

    if vals is not None:
        if vals.dtype.name == "category":
            vals = vals.cat.codes
        df_dict["x"] = vals

    df_sub = pd.DataFrame(df_dict).sort_values(["group", "time"])

    assert df_sub["time"].notna().all()
    assert df_sub.index.is_unique

    if vals is not None:
        agg_inst = agg(df_sub["x"].astype("float64").values)
    else:
        agg_inst = agg(len(groups))

    result = groupby_window_agg(
        df_sub["group"].values,
        df_sub["time"].astype("float64").values,
        agg_inst,
        start,
        end,
        store_flags=store_flags.values,
    )

    return pd.Series(result, index=df_sub.index)


def pd_groupby_expanding_agg(groups, agg, vals=None, store_flags=None):
    groups = cat_factorize(groups)

    df_dict = {"group": groups}

    if vals is not None:
        if vals.dtype.name == "category":
            vals = vals.cat.codes
        df_dict["x"] = vals

    df_sub = pd.DataFrame(df_dict).sort_values("group")

    assert df_sub.index.is_unique

    if vals is not None:
        agg_inst = agg(df_sub["x"].astype("float64").values)
    else:
        agg_inst = agg(len(groups))

    result = groupby_expanding_agg(df_sub["group"].values, agg_inst, store_flags=store_flags.values)

    return pd.Series(result, index=df_sub.index)


if __name__ == "__main__":
    import numpy as np

    gr = [1, 1, 1, 2, 2]
    t = [1, 3, 3, 4, 5]
    vals = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    # agg = SumAgg(vals)

    # groupby_window_agg(gr, t, agg, -1)

    # print(agg.result)

    # res = pd_groupby_window_agg(gr, t, SumAgg, vals, -1)

    res = pd_groupby_expanding_agg(gr, SumAgg, vals)

    print(res)
