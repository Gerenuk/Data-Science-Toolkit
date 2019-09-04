from functools import partial
from collections import deque, defaultdict
import copy

__author__ = "asuchane"


class SlidingWindowAggregator:
    def __init__(self, head_row_func, tail_row_func, aggregator, row_map_func=None):
        self.row_map_func = row_map_func
        self.tail_row_func = tail_row_func
        self.head_row_func = head_row_func
        self.aggregator = aggregator

    def __call__(self, row):
        head_rows = self.head_row_func(row)
        tail_rows = self.tail_row_func(row)

        if self.row_map_func is None:
            for tail_row in tail_rows:
                self.aggregator.remove_row(tail_row)
            for head_row in head_rows:
                self.aggregator.add_row(head_row)
        else:
            for tail_row in tail_rows:
                self.aggregator.remove_row(self.tail_row_func(tail_row))
            for head_row in head_rows:
                self.aggregator.add_row(self.head_row_func(head_row))

        return self.aggregator.value


def cur_row_head_func(row):
    yield row


class VarLagFunc:
    def __init__(self, val_func, lag_diff):
        self.lag_diff = lag_diff
        self.val_func = val_func
        self.row_cache = deque()

    def __call__(self, row):
        self.row_cache.append(row)
        cur_val = self.val_func(row)

        while self.val_func(self.row_cache[0]) < cur_val - self.lag_diff:
            yield self.row_cache.popleft()


class SumAggregator:
    def __init__(self, val_func):
        self.val_func = val_func
        self.value = 0

    def add_row(self, row):
        self.value += self.val_func(row)

    def remove_row(self, row):
        self.value -= self.val_func(row)


class MapVarAggregator:
    def __init__(self, key_func, aggregator_init_func):
        self.key_func = key_func
        self.key__aggregator = defaultdict(aggregator_init_func)
        self.value = None

    def add_row(self, row):
        aggregator_for_key = self.key__aggregator[self.key_func(row)]
        aggregator_for_key.add_row(row)
        self.value = aggregator_for_key.value

    def remove_row(self, row):
        self.key__aggregator[self.key_func(row)].remove_row(row)


if __name__ == "__main__":
    from pprint import pprint
    from operator import itemgetter as ig
    from operator import attrgetter as ag
    from collections import namedtuple
    import string

    T = namedtuple("Row", "time id val")
    data = [
        T(i, string.ascii_letters[t // 10 - 1], t % 10)
        for i, t in enumerate([11, 21, 12, 22, 23, 24, 15, 25])
    ]

    sliding_window = SlidingWindowAggregator(
        VarLagFunc(ag("time"), 7),
        VarLagFunc(ag("time"), 28),
        MapVarAggregator(ag("id"), partial(SumAggregator, ag("val"))),
    )

    # pprint(data)

    for d in data:
        print(d, "->", sliding_window(d), ";")
