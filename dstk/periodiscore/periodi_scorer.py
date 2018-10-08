__author__ = "Anton Suchaneck"
__email__ = "a.suchaneck@gmail.com"


class PeriodiScorer:
    """
    Summarizes a PeriodiBin into a fixed score by flattening the `period` dimension.
    Should only be called with points which are already in correct bin!
    (this is done automatically from PeriodiPredict)
    That is why a fixed pre-defined score is returned for any point.
    """
    __slots__ = ["_bin_score", "name", "period_scores", "periods"]

    def __init__(self, periodi_bin):
        self._bin_score = periodi_bin.score
        self.name = periodi_bin.name
        self.period_scores = periodi_bin.period_scores
        self.periods = frozenset(self.period_scores.keys())
        # not saving score_func. see __call__

    def __call__(self, point=None):
        return self._bin_score  # assumes that is already in correct bin

    def __repr__(self):
        return "[Bin {} : {:.2f}]".format(self.name, self._bin_score)
