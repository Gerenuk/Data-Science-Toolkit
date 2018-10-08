"""
TODO:
* make internal functions private (underscore)
* fix overlap_scores
* return period_num deaggregation information too?
* update docs
* convenience print function
* multiple periodicities possible?
* check all self. variables on combine/add
* aggregated score count is second stage?
* what if result list empty?
* time back-transformation?
* does shiftnum have same meaning throughout (single/double interval)?
"""


__author__ = "Anton Suchaneck"
__email__ = "a.suchaneck@gmail.com"


class PeriodiTrain:
    """
    PeriodiTrain is the first object to be initialized.
    It can add points through .add(point)
    """

    __slots__ = ["_bin_generator", "_bin_keys_func", "_bin_dict", "scorer_generator"]

    def __init__(self, bin_generator, bin_keys_func):
        """
        For PeriodiTrain.add(point) you need:
        >>> [(bin_key, data), ..] = self.bin_keys_func(point)
        >>> bin_obj = self.bin_generator(bin_key)
        >>> bin_obj.add(data)  # also returns self

        self.bin_dict = {bin_key: bin_obj, ..}
        """
        self._bin_generator = bin_generator
        self._bin_keys_func = bin_keys_func

        self._bin_dict = {}

    def add(self, point):
        for bin_key, data in self._bin_keys_func(point):
            # data (instead of point) needed for time period wrapping; so that no recalc
            if bin_key not in self._bin_dict:
                self._bin_dict[bin_key] = self._bin_generator(bin_key)
            self._bin_dict[bin_key].add(data)

            #if 0:
            #    score, period = self._bin_dict[bin_key]._score_func(data)
            #    print(">> {:%d.%m} {} into {} score {:.2f} period {}".format(point[0],
            #                                                                 point[1],
            #                                                                 bin_key,
            #                                                                 score,
            #                                                                 period,
            #                                                                 ))

        return self

    def merge(self, other):
        for key, other_bin in other.bin_dict.items():
            if key not in self._bin_dict:
                self._bin_dict[key] = other_bin
            else:
                self._bin_dict[key].merge(other_bin)
        return self

    @property
    def bin_dict(self):
        return self._bin_dict

    @property
    def bin_keys_func(self):
        return self._bin_keys_func
