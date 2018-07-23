Periodiscore
************

Cookbook
========

Training phase
--------------
You start by creating a "counter" called `PeriodiTrain` which is applied to all data sequentially.::

    from periodi_train import PeriodiTrain
    periodi_train = PeriodiTrain(bin_generator, bin_keys_func)

    for point in data:
        periodi_train.add(point)

This counter will accept data points for addition and internally create `PeriodiBin` bins as needed
with `bin_generator`. `bin_keys_func` is used to calculate the correct bin to be used in the dict.::

    from periodi_bin import PeriodiBin

    def bin_generator(bin_key):
        return PeriodiBin(score_func)


    def bin_keys_func(point):
        ...
        return [(bin_key, data), ...]   # later will use PeriodiBin.add(data) instead of point


    def score_func(data):
        ...
        return score, period


.. note::

    Therefore on `PeriodiTrain.add(point)`:

    #. calculate right bin through `bin_keys_func` (generate new bin by `bin_generator(bin_key)` if needed)
    #. at the same time transform `point` to `data`
    #. for all relevant bins call `PeriodiBin.add(data)` (`PeriodiBin` should only receive points which are already in the correct bin)

        #. calculate `score` and `period` through `PeriodiBin._score_func(data)`
        #. the bin-period score will be updated with `PeriodiBin._period_agg_func` (usually `max`)

`PeriodiBin._period_scores` is a dict which stores one score for each period `{period:score, ...}`.
You can use `PeriodiTrain.merge` to merge with other sub-counters - for example in Spark aggregate.

`PeriodiBin` is initialized by `bin_generator` to have `centre = bin + binnum_width / 2`.

Prediction phase
----------------
When all data points have been added, you generate a `PeriodiPredict` object. This is needed for pre-aggregating
bin results and also for separation of concerns.::

    from periodi_predict import PeriodiPredict
    predictor = PeriodiPredict(periodi_train, scorer_generator)

    point_score = predictor(point)

    for scorer, score, key in predictor.get_top(<exclude_bin_func>):
        ....


The `scorer_generator` function takes a `PeriodiBin` and returns a scorer function which will be called on points.
Note that the `periodi_train.bin_keys_func` will still be used to pre-calculate the possible bins.::

    def scorer_generator(periodi_bin):
        return PeriodiScorer(periodi_bin)

The scores of all scorer are aggregated by `PeriodiPredict._score_agg` (usually `max`).