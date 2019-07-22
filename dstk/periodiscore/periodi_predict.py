from operator import itemgetter


__author__ = "Anton Suchaneck"
__email__ = "a.suchaneck@gmail.com"


class PeriodiPredict:
    def __init__(self, periodi_train, scorer_generator, exclude_bin_func):
        """
        periodi_train.bin_dict = {key: bin_obj, ...}
        periodi_train.bin_keys_func(point) = [(key1, data), ...]
        self.scorer_generator(bin_obj)(data) -> score
        final_score=self._score_agg(scorer_scores)
        """
        self._scorers = {
            key: scorer_generator(bin_obj)
            for key, bin_obj in periodi_train.bin_dict.items()
        }
        self._bin_keys_func = periodi_train.bin_keys_func
        self._score_agg = max
        self._exclude_bin_func = exclude_bin_func

    def score(self, point):
        """
        Calls all scorers in appropriate bins
        """
        bin_keys = self._bin_keys_func(point)
        return self._score_agg(
            self._scorers[bin_key](data) for bin_key, data in bin_keys
        )

    def __repr__(self):
        return "\n".join(
            map(
                str,
                sorted(
                    self._scorers.values(), key=lambda x: x._bin_score, reverse=True
                ),
            )
        )

    def get_top(self):
        result_dict = {
            key: (scorer, scorer(), key) for key, scorer in self._scorers.items()
        }
        result = []
        while result_dict:
            best_scorer, best_score, best_key = max(
                result_dict.values(), key=itemgetter(1)
            )
            result.append((best_key, best_scorer, best_score))
            for key in self._exclude_bin_func(best_key):
                if key in result_dict:
                    del result_dict[key]
        return result
