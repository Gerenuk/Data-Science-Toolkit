from cmath import exp, phase
from collections import namedtuple
from datetime import timedelta
from functools import partial
from itertools import product
from math import floor, pi

from individual.anton.periodiscore.periodi_bin import PeriodiBin
from individual.anton.periodiscore.periodi_predict import PeriodiPredict
from individual.anton.periodiscore.periodi_scorer import PeriodiScorer
from individual.anton.periodiscore.periodi_train import PeriodiTrain

__author__ = "Anton Suchaneck"
__email__ = "a.suchaneck@gmail.com"

PointData = namedtuple("PointData", "amount_steps time_steps point")

days_in_month = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def float_to_date(float_date, month):
    days_in_this_month = days_in_month[month]
    day = float_date * days_in_this_month + 1
    if day < 1:
        return 1
    if day > days_in_this_month:
        return days_in_this_month
    return round(day)


def triangle_product_score(
    data,
    amount_centre,
    amount_zero,
    amount_cap,
    time_centre,
    time_cap,
    time_zero,
    time_step_num,
    power=1,
):
    """
    ranges are:
    centre = 1
    centre+-cap = 1
    and linear interpolation to
    centre+-zero = 0

    all parameters are in terms of stepsize
    gets data from bin_keys_func
    """
    amount_in_steps, time_in_steps, _point = data

    # amount score
    amount_dist = abs(amount_in_steps - amount_centre)
    assert amount_dist <= amount_zero, (amount_dist, amount_zero)

    if amount_dist <= amount_cap:
        amount_score = 1.0
    else:
        amount_score = pow(
            1.0 - (amount_dist - amount_cap) / (amount_zero - amount_cap), power
        )

    # time score
    bin_time_start = floor(time_in_steps / time_step_num) * time_step_num + time_centre
    bin_time_start2 = bin_time_start - time_step_num

    time_dist = abs(time_in_steps - bin_time_start)
    time_dist2 = abs(time_in_steps - bin_time_start2)

    if time_dist2 < time_dist:
        time_dist = time_dist2
        bin_time_start = bin_time_start2

    period = floor(bin_time_start / time_step_num)

    assert time_dist <= time_zero, (time_dist, time_dist2, time_zero)

    if time_dist <= time_cap:
        time_score = 1.0
    else:
        time_score = pow(1.0 - (time_dist - time_cap) / (time_zero - time_cap), power)

    # total score
    score = float(amount_score * time_score)  # float needed for spark
    assert score >= 0, score

    return score, period


def exclude_bin_func(bin, *, time_step_num, amount_binnum_width, time_binnum_width):
    amount_bin, time_bin = bin
    return [
        (amount_bin + amount_offset, (time_bin + time_offset) % time_step_num)
        for amount_offset, time_offset in product(
            range(-amount_binnum_width, amount_binnum_width + 1),
            range(-time_binnum_width, time_binnum_width + 1),
        )
    ]


def date_to_float(date):
    return date.year * 12 + date.month + (date.day - 1) / days_in_month[date.month]


def date_to_float_weekend_fix(date):
    if date.weekday() == 0:
        date - timedelta(1)
    return date_to_float(date)


def periodi_amount_time(
    points,
    *,
    amount_binnum_width,
    amount_cap=0.5,
    amount_step,
    time_binnum_width,
    time_cap=0.5,
    time_step_num,
    date_to_float=date_to_float,
    power=1
):
    periodi_train = PeriodiTrain(
        bin_generator(
            amount_binnum_width=amount_binnum_width,
            amount_cap=amount_cap,
            time_step_num=time_step_num,
            time_binnum_width=time_binnum_width,
            time_cap=time_cap,
            amount_step=amount_step,
            power=power,
        ),
        partial(
            bin_keys_func,
            amount_step=amount_step,
            amount_binnum_width=amount_binnum_width,
            time_step_num=time_step_num,
            time_binnum_width=time_binnum_width,
            date_to_float=date_to_float,
        ),
    )

    exclude_bin_func_partial = partial(
        exclude_bin_func,
        time_step_num=time_step_num,
        amount_binnum_width=amount_binnum_width,
        time_binnum_width=time_binnum_width,
    )

    for point in points:
        periodi_train.add(point)

    return PeriodiPredict(periodi_train, PeriodiScorer2, exclude_bin_func_partial)


def bin_keys_func(
    point,
    *,
    amount_step,
    amount_binnum_width,
    time_step_num,
    time_binnum_width,
    date_to_float
):
    """
    should return a tuple (bin, data) where data will be passed to the score_func
    """
    time, amount = point

    amount_in_steps = amount / amount_step
    amount_bin_base = floor(amount_in_steps)

    time_step = 1 / time_step_num

    time_in_steps = date_to_float(time) / time_step
    time_bin_base = floor(time_in_steps) % time_step_num

    result = []
    for amount_offset, time_offset in product(
        range(amount_binnum_width), range(time_binnum_width)
    ):
        time_bin = time_bin_base - time_offset
        amount_bin = amount_bin_base - amount_offset

        if time_bin < 0:
            time_bin += time_step_num

        result.append(
            (
                (amount_bin, time_bin),  # bin
                PointData(
                    amount_in_steps,  # data: what score_func gets (i.e. pre-processed)
                    time_in_steps,
                    point,
                ),
            )
        )

    return result


def bin_generator(
    *,
    amount_binnum_width,
    amount_cap=0.5,
    amount_step,
    time_binnum_width,
    time_cap=0.5,
    time_step_num,
    power
):
    amount_zero = amount_binnum_width / 2
    time_zero = time_binnum_width / 2

    def bin_generator_wrapped(bin_key):
        """
        generates bins when needed for a new bin_key
        """
        amount_bin, time_bin = bin_key
        amount_centre = amount_bin + amount_zero  # this sets the centre of the bin
        time_centre = time_bin + time_zero
        return PeriodiBin2(
            score_func=partial(
                triangle_product_score,
                amount_centre=amount_centre,
                amount_cap=amount_cap,
                amount_zero=amount_zero,
                time_centre=time_centre,
                time_cap=time_cap,
                time_zero=time_zero,
                time_step_num=time_step_num,
                power=power,
            ),
            time_step_num=time_step_num,
            name="a{} t{}".format(amount_centre * amount_step, time_centre),
        )

    return bin_generator_wrapped


class PeriodiScorer2(PeriodiScorer):
    def __init__(self, periodi_bin):
        super().__init__(periodi_bin)

        self.point_cnt = periodi_bin.point_cnt

        self.avg_amount = periodi_bin.amount_sum / periodi_bin.point_cnt

        time_sum = periodi_bin.time_sum
        time_pos = phase(time_sum) / pi / 2  # however will return 0 when ambigious
        if time_pos < 0:
            time_pos += 1
        self.avg_time = time_pos


class PeriodiBin2(PeriodiBin):
    __slots__ = PeriodiBin.__slots__ + [
        "amount_sum",
        "time_sum",
        "point_cnt",
        "time_step_num",
    ]

    def __init__(self, score_func, time_step_num, name=""):
        super().__init__(score_func, name)
        self.amount_sum = 0
        self.time_sum = 0
        self.point_cnt = 0
        self.time_step_num = time_step_num

    def add(self, data):
        super().add(data)

        amount_in_steps, time_in_steps, point = data
        time, amount = point

        self.amount_sum += amount
        self.time_sum += exp(
            2j * pi * (time_in_steps % self.time_step_num) / self.time_step_num
        )
        self.point_cnt += 1
