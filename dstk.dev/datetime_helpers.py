import calendar
import datetime as dt
from functools import lru_cache
from math import modf
import numpy as np

from cytoolz import concat
from pandas.tseries.offsets import CDay

HOLIDAYS = (list(concat([(dt.date(year, 1, 1),
                          dt.date(year, 5, 1),
                          dt.date(year, 10, 3),
                          dt.date(year, 12, 24),
                          dt.date(year, 12, 25),
                          dt.date(year, 12, 26),
                          dt.date(year, 12, 31),
                          ) for year in range(2010, 2020)])) +
            [dt.date(2014, 4, 18), dt.date(2015, 4, 3)] +  # Karfreitag
            [dt.date(2014, 4, 21), dt.date(2015, 4, 5)] +  # Ostermontag
            [dt.date(2014, 5, 29), dt.date(2015, 5, 14)] +  # Christi Himmelfahrt
            [dt.date(2014, 6, 9), dt.date(2015, 5, 25)]  # Pfingstmontag
            )

busday_offset = CDay(holidays=HOLIDAYS)


def add_month(year, month, delta_month):
    month += delta_month
    while month > 12:
        month -= 12
        year += 1
    while month < 1:
        month += 12
        year -= 1
    return year, month


def busday_forward(date):
    return busday_offset.rollforward(date).to_pydatetime().date()


def busday_back(date):
    return busday_offset.rollback(date).to_pydatetime().date()


@lru_cache(maxsize=1000)
def busday_diff(date1, date2):
    return int(np.busday_count(date1, date2, busdaycal=busday_offset.calendar))


@lru_cache(maxsize=300)
def busday_diff_month_start(date):
    return busday_diff(dt.datetime(date.year, date.month, 1), date)


def busday_diff_month_end(date):
    return busday_diff(date, last_day_of_month(date.year, date.month))


@lru_cache(maxsize=300)
def busday_in_month(year, month, busday):
    return (busday_offset.rollforward(dt.datetime(year, month, 1)) + busday * busday_offset).to_pydatetime().date()


def last_day_of_month(year, month):
    return dt.date(year, month, calendar.monthrange(year, month)[1])


@lru_cache(maxsize=300)
def days_in_month(year, month):
    return calendar.monthrange(year, month)[1]


@lru_cache(maxsize=300)
def busdays_in_month(year, month):
    first_of_month = dt.date(year, month, 1)
    next_year, next_month = add_month(year, month, 1)
    next_month = dt.date(next_year, next_month, 1)
    return busday_diff(first_of_month, next_month)


def date_info(date):
    return "s{:02} e{:02} w{}".format(
        np.busday_count(dt.date(date.year, date.month, 1), date),
        np.busday_count(date, last_day_of_month(date.year, date.month)),
        date.weekday(),
    )


@lru_cache(maxsize=300)
def date_to_float(date):
    return date.year * 12 + (date.month - 1) + (date.day - 1) / days_in_month(date.year, date.month)


@lru_cache(maxsize=300)
def busdate_to_float(date):
    return date.year * 12 + (date.month - 1) + busday_diff_month_start(date) / busdays_in_month(date.year, date.month)


def float_to_date(float_date):
    day_float, year_month = modf(float_date)
    year_month = int(year_month)
    year = year_month // 12
    month = (year_month % 12) + 1

    days_in_this_month = days_in_month(year, month)
    day = round(day_float * days_in_this_month + 1)

    if day < 1:
        day = 1
    elif day > days_in_this_month:
        day = days_in_this_month

    return dt.date(year, month, day)


def float_to_busdate(float_date):
    day_float, year_month = modf(float_date)
    year_month = int(year_month)
    year = year_month // 12
    month = (year_month % 12) + 1

    busdays_in_this_month = busdays_in_month(year, month)
    busday = round(day_float * busdays_in_this_month)

    if busday > busdays_in_this_month:
        busday = busdays_in_this_month

    return busday_in_month(year, month, busday)


if __name__ == '__main__':
    from dateutil.relativedelta import relativedelta
    import numpy as np

    # print(busday_forward(dt.datetime(2016, 1, 1)))
    # print(busday_back(dt.datetime(2016, 1, 2)))
    # print(last_day_of_month(2016, 1))
    # print(busdays_in_month(2014, 1))
    # for i in range(1, 32):
    #    print(busday_diff_month_start(dt.date(2014, 1, i)))

    d = dt.date(2014, 2, 3)
    print(d)
    f = date_to_float(d)
    print(f)
    d2 = float_to_date(f)
    print(d2)

    for d in range(1000):
        d1 = dt.date(2014, 1, 1) + relativedelta(days=d)
        f = busdate_to_float(d1)
        d2 = float_to_busdate(f)
        if np.is_busday(d1, busdaycal=busday_offset.calendar) and d1 != d2:
            print("ERROR", d1, d2)
            break
