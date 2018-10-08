import calendar
import datetime as dt

import numpy as np
from individual.anton.periodiscore.periodi_date_amount import *

__author__ = "Anton Suchaneck"
__email__ = "a.suchaneck@gmail.com"

HOLIDAYS = [dt.datetime(year, month, day)
            for year in [2016, 2017]
            for day, month in [(1, 1), (25, 12), (26, 12), ]]


def create_day_float_map(cnt_daydiff, start_date, holidays):
    busdaycal = np.busdaycalendar(holidays=holidays)

    result = {}
    for daydiff in range(cnt_daydiff):
        date = start_date + dt.timedelta(daydiff)
        first_day_in_month = date.replace(day=1)
        last_day_in_month = date.replace(day=calendar.monthrange(date.year, date.month)[1])
        max_cnt_busdays = np.busday_count(first_day_in_month, last_day_in_month, busdaycal=busdaycal)
        cnt_busdays = np.busday_count(first_day_in_month, date, busdaycal=busdaycal)
        result[date] = 12 * date.year + (date.month - 1) + float(cnt_busdays / (max_cnt_busdays + 1))

    return result


busday_float = create_day_float_map(cnt_daydiff=1000,
                                    start_date=dt.datetime(2016, 1, 1),
                                    holidays=HOLIDAYS
                                    ).get


def npdatetime2datetime(npdt):
    return dt.datetime.utcfromtimestamp((npdt - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s"))
    # return dt.datetime.utcfromtimestamp(npdt.astype(int))


def create_yearmonth_func(years, holidays):
    def yearmonth_func(year, month, day_float):
        max_cnt_busdays = yearmonth__max_cnt_busdays[year, month]
        target_bday = int(round(day_float * (max_cnt_busdays + 1)))  # int needed since round(np.float64) yields float
        first_day_in_month = dt.datetime(year, month, 1)

        return npdatetime2datetime(np.busday_offset(first_day_in_month,
                                                    target_bday,
                                                    "forward",
                                                    busdaycal=busdaycal,
                                                    ))

    busdaycal = np.busdaycalendar(holidays=holidays)

    yearmonth__max_cnt_busdays = {}

    for year in years:
        for month in range(1, 13):
            first_day_in_month = dt.datetime(year, month, 1)
            last_day_in_month = dt.datetime(year, month, calendar.monthrange(year, month)[1])
            max_cnt_busdays = np.busday_count(first_day_in_month, last_day_in_month, busdaycal=busdaycal)
            yearmonth__max_cnt_busdays[year, month] = int(max_cnt_busdays)

    return yearmonth_func


float_busdate = create_yearmonth_func(years=[2016, 2017],
                                      holidays=HOLIDAYS)


def futuremoney(data):
    """
    :param data: [(datetime, amount), ...]
    :result: future (datetime, amount)
    """
    if len(data) > 15:
        return None, None

    months = None

    periodi = periodi_amount_time(data,
                                  amount_binnum_width=1,
                                  amount_cap=0.5,
                                  time_step_num=30,
                                  time_binnum_width=10,
                                  time_cap=0.5,
                                  amount_step=49,
                                  power=1,
                                  date_to_float=date_to_float
                                  )
    tops = periodi.get_top()

    if tops:
        key, scorer, score = tops[0]
        return float_to_date(scorer.avg_time), scorer.avg_amount
    else:
        return None, None


if __name__ == '__main__':
    for d in range(60):
        date = dt.datetime(2016, 1, 1) + dt.timedelta(days=d)
        f = day_float(date)
        print(date, f, float_date(date.year, date.month, f))
