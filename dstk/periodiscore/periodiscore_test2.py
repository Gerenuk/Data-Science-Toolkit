import datetime as dt
import logging
from collections import namedtuple

logging.basicConfig()
logg = logging.getLogger()
logg.setLevel(logging.INFO)

from individual.anton.periodiscore.periodi_date_amount import (
    periodi_amount_time,
    float_to_date,
)

TimeAmount = namedtuple("TimeAmount", "time amount")

data1 = [
    (20160129, -524.07),
    (20160229, -524.07),
    (20160330, -524.07),
    (20160429, -524.07),
    # (20160530, -524.07 ),
    # (20160630, -524.07 ),
]

data2 = [
    (20160104, -14.0),
    (20160201, -14.0),
    (20160301, -14.0),
    (20160401, -14.0),
    # (20160502, -14.0 ),
    # (20160601, -14.0 ),
    # (20160701, -14.0 ),
]

data = []
for date, amount in data2:
    date = str(date)
    data.append(
        TimeAmount(dt.datetime(int(date[:4]), int(date[4:6]), int(date[6:8])), amount)
    )

periodi = periodi_amount_time(
    data,
    amount_binnum_width=1,
    amount_cap=0.5,
    time_step_num=30,
    time_binnum_width=10,
    time_cap=0.5,
    amount_step=49,
    power=3,
)

print("--- Top bins ---")
for best_key, best_scorer, best_score in periodi.get_top():
    print(
        "Amount={scorer.avg_amount} Day={day} Periods=({periods}) Score={score:.1f}".format(
            scorer=best_scorer,
            day=float_to_date(best_scorer.avg_time),
            periods=", ".join(map(str, sorted(best_scorer.periods))),
            score=best_score,
        )
    )

print("--- Data scores ---")
for point in data:
    print(
        "{} {}€ -> {:.2f}".format(
            point.time.strftime("%d.%m.%y"), point.amount, periodi.score(point)
        )
    )

from pprint import pprint

pprint(periodi._scorers)
