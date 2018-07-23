# -*- coding: utf-8 -*-
import itertools as itoo
import datetime as dt
from datetime import date, timedelta
import subprocess


def week_start_date(year, week):
    d = date(year, 1, 1)
    delta_days = d.isoweekday() - 1
    delta_weeks = week
    if year == d.isocalendar()[0]:
        delta_weeks -= 1
    delta = timedelta(days=-delta_days, weeks=delta_weeks)
    return d + delta


def num2date(dateymd):
    year = dateymd // 10000
    month = dateymd // 100 % 100
    day = dateymd % 100
    return dt.datetime(year, month, day)


def date_range(dateymd, months):  # negative months?
    year = dateymd // 10000
    month = dateymd // 100 % 100
    day = dateymd % 100

    for m in range(months + 1):
        yield year * 10000 + month * 100 + day
        month += 1
        if month > 12:
            month = 1
            year += 1
        elif month < 1:
            month = 12
            year -= 1


def date_add(dateymd, months):
    year = dateymd // 10000
    month = dateymd // 100 % 100
    day = dateymd % 100

    month += months

    year += (month - 1) // 12
    month = (month - 1) % 12 + 1

    return year * 10000 + month * 100 + day


def date_fix_ym(year, month):
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return year, month


class ColorMapper:
    def __init__(self, known_map=None, colors=None):
        from brewer2mpl import qualitative
        self.known_color_map = dict(known_map) if known_map is not None else {}

        self.prev_used_colors = {}
        self.color_set = colors or qualitative.Set1["max"].mpl_colors
        self.reset()

    def __getitem__(self, key):
        if key in self.known_color_map:
            return self.known_color_map[key]

        if key in self.assigned_colors:
            return self.assigned_colors[key]

        if key in self.prev_used_colors:
            old_color = self.prev_used_colors[key]
            if old_color not in self.assigned_colors.values():
                self._set_color(key, old_color)
                return old_color

        if self.color_pool:
            new_color = self.color_pool.pop()
            self._set_color(key, new_color)
            return new_color

        new_color = next(self.color_cycle)
        self._set_color(key, new_color)
        return new_color

    def _set_color(self, key, color):
        if key in self.assigned_colors:
            raise KeyError("Resetting color {} for key {}".format(color, key))
        if color in set(self.assigned_colors.values()) and self.color_pool:
            raise ValueError("Color pool not empty {} but color clash {}".format(self.color_pool, self.assigned_colors))
        self.assigned_colors[key] = color
        if key not in self.prev_used_colors:
            self.prev_used_colors[key] = color
        self.color_pool.discard(color)

    def reset(self):
        self.assigned_colors = {}
        self.color_pool = set(self.color_set)
        self.color_cycle = itoo.cycle(self.color_set)


class DefMap:
    def __init__(self, default, **default_map):
        self.default_map = default_map
        self.default_map["default"] = default

    def __call__(self, key="default"):
        if isinstance(key, str):
            return self.default_map[key]
        else:
            return key

# def date_range(year, month, endyear=None, endmonth=None, day=1):
#    if year2<year1 or (year1==year2 and month2<month1):
#        raise ValueError("Date range not ordered: {}.{} -> {}.{}".format(year1, month1, year2, month2))
#    year=year1
#    month=month1
#    while 1:
#        yield year*10000+month*100+day
#        if year==year2 and month==month2:
#            break
#        month+=1
#        if month>12:
#            month=1
#            year+=1



import io
import csv
import os


class ProgressCsv:
    def __init__(self, filename, **kwargs):
        self.file = open(filename, "rb")
        text_file = io.TextIOWrapper(self.file, newline='')
        self.reader = csv.reader(text_file, **kwargs)
        self.total_size = os.path.getsize(filename)

    def __iter__(self):
        for line in self.reader:
            yield line

    def __len__(self):
        return os.path.getsize(self.file.name)

    @property
    def iterations(self):
        return self.file.tell()


def progress_csv(*args, **kwargs):
    from tqdm import tqdm
    return tqdm(ProgressCsv(*args, **kwargs))


def copy_file(source, dest):
    subprocess.call(['xcopy', source, dest, "/Y"])   # option for overwrite without asking