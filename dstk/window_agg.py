import numba


@numba.jitclass([("cur_value", numba.float64)])
class SumAgg:
    def __init__(self):
        self.cur_value = 0

    def add(self, val):
        # print(f"Adding {val}")
        self.cur_value += val

    def remove(self, val):
        # print(f"Removing {val}")
        self.cur_value -= val

    def reset(self):
        self.cur_value = 0
        
    def value(self):
        return self.cur_value
        
        
@numba.jitclass([("value", [])])
class MaxAgg:
    def __init__(self):
        self.values = []

    def add(self, val):
        # print(f"Adding {val}")
        self.values.append(val)

    def remove(self, val):
        # print(f"Removing {val}")
        self.cur_value -= val

    def reset(self):
        self.values = []
        
    def value(self):
        return self.cur_value
        
        
@numba.jitclass([("cur_value", numba.float64)])
class CountAgg:
    def __init__(self):
        self.cur_value = 0

    def add(self, val):
        self.cur_value += 1

    def remove(self, val):
        self.cur_value -= 1

    def reset(self):
        self.cur_value = 0
        
    def value(self):
        return self.cur_value
        
        
@numba.jitclass([("cnts", dict())])
class NuniqueAgg:
    def __init__(self):
        self.cnts = {}
        
    def add(self, val):
        # print(f"Adding {val}")
        if val in self.cnts:
            self.cnts[val]+=1
        else:
            self.cnts[val]=1

    def remove(self, val):
        # print(f"Removing {val}")
        if self.cnts[val]==1:
            del self.cnts[val]
        else:
            self.cnts[val]-=1

    def reset(self):
        self.cur_set = {}
        
    def value(self):
        return len(self.cnts)


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

        print(f"[{cur_idx}] {vals[cur_idx]} gr {group[cur_idx]}: {start_idx}-{end_idx} -> {aggregator.value()}")

        # Store current aggregation result
        result[cur_idx] = aggregator.value()

    return result


def pd_groupby_window_agg(df, group, time, val, agg, window_start, window_end=0):
    df_sub = df[[group, time, val]].sort_values([group, time])

    agg_vals = groupby_window_agg(
        df_sub[group].factorize()[0],
        df_sub[time].values,
        df_sub[val].values,
        agg,
        window_start,
        window_end,
    )

    return pd.Series(agg_vals, index=df_sub.index)


if __name__=="__main__":
    res = pd_groupby_window_agg(dd, "gr", "t", "x", NuniqueAgg(), -1, 2)
    dd["agg"] = res
    dd.sort_values(["gr", "t"])