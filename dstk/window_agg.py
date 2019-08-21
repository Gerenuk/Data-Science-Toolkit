import numba
import numpy as np
import pandas as pd
from functools import reduce
from operator import add


group_sep = "-"


@numba.jitclass([("cur_value", numba.float64)])
class SumAgg:
    def __init__(self):
        self.cur_value = 0

    def add(self, val):
        self.cur_value += val

    def remove(self, val):
        self.cur_value -= val

    def reset(self):
        self.cur_value = 0

    def value(self):
        return self.cur_value


@numba.jitclass([("cur_value", numba.float64)])
class CountAgg:
    def __init__(self):
        self.cur_value = 0

    def add(self, val):
        """
        Ignores val
        """
        self.cur_value += 1

    def remove(self, val):
        self.cur_value -= 1

    def reset(self):
        self.cur_value = 0

    def value(self):
        return self.cur_value


@numba.jitclass([("cnts", numba.types.DictType(numba.float64, numba.int64))])
class NuniqueAgg:
    def __init__(self):
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )

    def add(self, val):
        if val in self.cnts:
            self.cnts[val] += 1
        else:
            self.cnts[val] = 1

    def remove(self, val):
        if self.cnts[val] == 1:
            del self.cnts[val]
        else:
            self.cnts[val] -= 1

    def reset(self):
        self.cnts = numba.typed.Dict.empty(
            key_type=numba.float64, value_type=numba.int64
        )

    def value(self):
        return len(self.cnts)


@numba.jitclass([("last_value", numba.float64), ("cur_value", numba.float64)])
class DiffAgg:
    def __init__(self):
        self.last_value = np.nan
        self.cur_value = np.nan

    def add(self, val):
        self.last_value = self.cur_value
        self.cur_value = val

    def reset(self):
        self.last_value = np.nan
        self.cur_value = np.nan

    def value(self):
        return self.cur_value - self.last_value


@numba.jit(nogil=True, nopython=True)
def groupby_window_agg(
    group, time_vals, vals, aggregator, timediff_start, timediff_end=0
):
    N = vals.size

    result = np.zeros(N, dtype=np.float64)  # result currently only float64 array

    start_idx = 0
    end_idx = 0

    cur_group = group[0]
    aggregator.add(vals[0])

    for cur_idx in range(len(group)):
        # Track group variable
        new_group = group[cur_idx]

        if new_group != cur_group:
            aggregator.reset()

            start_idx = cur_idx
            end_idx = cur_idx

            cur_group = new_group
            aggregator.add(vals[cur_idx])

        cur_time = time_vals[cur_idx]

        # print(
        #    f"// [i:{cur_idx}, t:{time_vals[cur_idx]}] gr:{group[cur_idx]}, x:{vals[cur_idx]}; before {start_idx}-{end_idx}"
        # )

        # Adjust window end
        if timediff_end != 0:
            window_end_time = cur_time + timediff_end

            while end_idx < N - 1 and group[end_idx + 1] == cur_group:
                if time_vals[end_idx + 1] > window_end_time:
                    break

                end_idx += 1
                aggregator.add(vals[end_idx])

        elif end_idx < N - 1 and group[end_idx + 1] == cur_group:
            if end_idx < cur_idx:  # otherwise at start of a group and already added
                aggregator.add(vals[end_idx])

            end_idx = cur_idx

        # Adjust window start
        if timediff_start != 0:
            window_start_time = cur_time + timediff_start

            while start_idx < N - 1:
                if time_vals[start_idx] >= window_start_time:
                    break

                aggregator.remove(vals[start_idx])
                start_idx += 1

        elif start_idx < N - 1 and group[start_idx + 1] == cur_group:
            if start_idx < cur_idx:
                aggregator.remove(vals[start_idx])

            start_idx = cur_idx

        # print(
        #    f"-> {start_idx} - {end_idx} -> {aggregator.value()}"
        # )

        # Store current aggregation result
        result[cur_idx] = aggregator.value()

    return result


@numba.jit(nogil=True, nopython=True)
def groupby_window_count(group, time_vals, timediff_start, timediff_end=0):
    N = vals.size

    result = np.zeros(N, dtype=np.float64)  # result currently only float64 array

    start_idx = 0
    end_idx = 0

    cur_group = group[0]

    for cur_idx in range(len(group)):
        # Track group variable
        new_group = group[cur_idx]

        if new_group != cur_group:
            start_idx = cur_idx
            end_idx = cur_idx

            cur_group = new_group

        cur_time = time_vals[cur_idx]

        # Adjust window end
        if timediff_end != 0:
            window_end_time = cur_time + timediff_end

            while end_idx < N - 1 and group[end_idx + 1] == cur_group:
                if time_vals[end_idx + 1] > window_end_time:
                    break

                end_idx += 1

        elif end_idx < N - 1 and group[end_idx + 1] == cur_group:
            end_idx = cur_idx

        # Adjust window start
        if timediff_start != 0:
            window_start_time = cur_time + timediff_start

            while start_idx < N - 1:
                if time_vals[start_idx] >= window_start_time:
                    break

                start_idx += 1

        elif start_idx < N - 1 and group[start_idx + 1] == cur_group:
            start_idx = cur_idx

        # Store current aggregation result
        result[cur_idx] = end_idx - start_idx + 1

    return result


@numba.jit(nogil=True, nopython=True)
def groupby_expanding_agg(group, vals, aggregator):
    N = vals.size

    result = np.zeros(N, dtype=np.float64)  # result currently only float64 array

    cur_group = group[0]

    for cur_idx in range(len(group)):
        # Track group variable
        new_group = group[cur_idx]

        if new_group != cur_group:
            aggregator.reset()
            cur_group = new_group

        aggregator.add(vals[cur_idx])

        # print(vals[cur_idx], "->", aggregator.last_value, aggregator.cur_value)

        # Store current aggregation result
        result[cur_idx] = aggregator.value()

    return result


def _group_col(groups):
    if isinstance(groups, pd.DataFrame):
        groups = groups.astype(str).pipe(
            lambda d: reduce(add, [col + group_sep for name, col in d.iteritems()])
        )

    groups = groups.factorize()[0]

    return groups


def pd_groupby_expanding_agg(groups, time, val, agg):
    """
    :param group: DataFrame
    :param time: time values (numbers)
    :param val: values to aggregate
    """
    groups = _group_col(groups)

    if val.dtype.name == "category":
        val = val.cat.codes

    df_sub = pd.DataFrame({"group": groups, "time": time, "x": val}).sort_values(
        ["group", "time"]
    )

    assert df_sub.index.is_unique

    agg_vals = groupby_expanding_agg(
        df_sub["group"].values, df_sub["x"].astype("float64").values, agg
    )

    return pd.Series(agg_vals, index=df_sub.index)


def pd_groupby_window_count(groups, time, window_start, window_end=0):
    groups = _group_col(groups)

    df_sub = pd.DataFrame({"group": groups, "time": time}).sort_values(
        ["group", "time"]
    )

    assert df_sub.index.is_unique

    agg_vals = groupby_window_count(
        df_sub["group"].values, df_sub["time"].values, window_start, window_end
    )

    return pd.Series(agg_vals, index=df_sub.index)


def pd_groupby_window_agg(groups, time, val, agg, window_start, window_end=0):
    """
    :param group: DataFrame
    :param time: time values (numbers)
    :param val: values to aggregate
    """
    groups = _group_col(groups)

    if val.dtype.name == "category":
        val = val.cat.codes

    df_sub = pd.DataFrame({"group": groups, "time": time, "x": val}).sort_values(
        ["group", "time"]
    )

    assert df_sub.index.is_unique

    agg_vals = groupby_window_agg(
        df_sub["group"].values,
        df_sub["time"].values,
        df_sub["x"].astype("float64").values,
        agg,
        window_start,
        window_end,
    )

    return pd.Series(agg_vals, index=df_sub.index)
