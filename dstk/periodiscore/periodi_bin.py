from collections import defaultdict

__author__ = "Anton Suchaneck"
__email__ = "a.suchaneck@gmail.com"


class PeriodiBin:
    """
    PeriodiBin is sitting at a particular bin_key.
    It can add data points through `.add(data)` which will
    * calculate the score with `.score_func(data)` which returns score and period
    * update score of `.period_scores[period]` to `period_agg_func(old_score, new_score)`

    The score needs to interact with `.period_agg_func`

    It offers a `pbin1.merge(pbin2)` operation which can be used in reduce/combine.
    """

    __slots__ = ["_score_func", "_period_agg_func", "_period_scores", "name"]

    def __init__(self, score_func, name=""):
        """
        :param score_func: such that `score, period = self.score_func(point)`

        .period_scores = {period: score, ...}
        .period_agg_func: How to aggregate scores of point at same period (e.g. max or sum)
        """
        self._score_func = score_func
        self.name = name

        self._period_agg_func = max  #
        self._period_scores = defaultdict(
            float
        )  # the key is the period of the data point

    def add(self, data):
        """
        The data point should be recognized by self.score_func
        `score, period = self.score_func(data)`
        `new_total_score=self.period_agg_func(old_score, new_score)`
        """
        new_score, period = self._score_func(data)
        self._period_scores[period] = self._period_agg_func(
            self._period_scores[period], new_score
        )

    @property
    def score(self):
        return sum(self._period_scores.values())

    @property
    def period_scores(self):
        return dict(self._period_scores)

    def __repr__(self):
        return "PeriodiBin {}({})".format(self.name, self._period_scores)

    def merge(self, other):
        for period, other_score in other._period_scores.items():
            self._period_scores[period] = self._period_agg_func(
                self._period_scores[period], other_score
            )
