from math import sqrt
from sklearn.base import clone
from sklearn.model_selection import cross_validate
import numpy as np
from operator import itemgetter
import time
from dstk.ml import color_score, color_param_val, color_param_name, format_if_number
import datetime as dt
import pytz


my_timezone = pytz.timezone(
    "Europe/Berlin"
)  # used for time output (e.g. on remote cloud server)


class SearchStop(Exception):
    pass


class GoldenSearch:
    """
    The decision to go left or right is made without noise

    def func(x):
        return x**2

    g=GoldenSearch(-10, 10)
    gen=g.val_gen()
    y=None
    try:
        while g.c-g.a>0.1:
            x = gen.send(y)
            y = func(x)
            print(g)
    except SearchStop as exc:
        pass
    """

    pos = 2 - (1 + sqrt(5)) / 2  # ~0.382

    def __init__(
        self,
        x0,
        x1,
        y0=np.nan,
        y1=np.nan,
        *,
        xm=np.nan,
        ym=np.nan,
        min_bound=True,
        max_bound=True,
        noise=0,
        map_value=None,
    ):
        if map_value is None:
            map_value = lambda x: x
        self.map_value = map_value

        self.a = map_value(x0)
        self.c = map_value(x1)

        if np.isnan(xm):
            xm = map_value(self.a + self.pos * (self.c - self.a))
        self.b = xm

        if min_bound is True:
            self.min_bound = self.a
        else:
            self.min_bound = min_bound

        if max_bound is True:
            self.max_bound = self.c
        else:
            self.max_bound = max_bound

        self.noise = noise

        self.ya = y0
        self.yb = ym
        self.yc = y1

        self.new_x = np.nan
        self.new_y = np.nan

    def _map_value(self, value):
        value = self.map_value(value)
        if value == self.a or value == self.b or value == self.c:
            raise SearchStop(f"Repeated value {value}")

        return value

    def val_gen(self):
        if np.isnan(self.ya):
            self.ya = yield self.a

        if np.isnan(self.yc):
            self.yc = yield self.c

        if np.isnan(self.yb):
            self.yb = yield self.b

        while 1:
            d1 = self.b - self.a
            d2 = self.c - self.b

            if self.ya < self.yb <= self.yc + self.noise:  # extend region left
                if self.min_bound < self.a:
                    self.new_x = self._map_value(self.b - d1 / self.pos)

                    if self.new_x < self.min_bound:
                        self.new_x = self.min_bound

                    self.new_y = yield self.new_x
                    self.a, self.b, self.c = self.new_x, self.a, self.b
                    self.ya, self.yb, self.yc = self.new_y, self.ya, self.yb
                else:
                    self.new_x = self._map_value(self.a + self.pos * (self.b - self.a))

                    self.new_y = yield self.new_x
                    self.b, self.c = self.new_x, self.b
                    self.yb, self.yc = self.new_y, self.yb

            elif self.ya + self.noise >= self.yb > self.yc:  # extend region right
                if self.c < self.max_bound:
                    self.new_x = self._map_value(self.b + d2 / self.pos)

                    if self.new_x > self.max_bound:
                        self.new_x = self.max_bound

                    self.new_y = yield self.new_x
                    self.a, self.b, self.c = self.b, self.c, self.new_x
                    self.ya, self.yb, self.yc = self.yb, self.yc, self.new_y
                else:
                    self.new_x = self._map_value(self.c - self.pos * (self.c - self.b))

                    self.new_y = yield self.new_x
                    self.a, self.b = self.b, self.new_x
                    self.ya, self.yb = self.yb, self.new_y

            elif self.ya >= self.yb - self.noise and self.yb - self.noise <= self.yc:
                if d1 < d2:
                    self.new_x = self._map_value(
                        self.c - (1 - self.pos) * (self.c - self.b)
                    )

                    self.new_y = yield self.new_x

                    if self.new_y > self.yc + self.noise:
                        raise SearchStop("Inconsistent y > c")

                    if self.new_y < self.yb:
                        self.a, self.b = self.b, self.new_x
                        self.ya, self.yb = self.yb, self.new_y
                    elif self.new_y > self.yb:
                        self.c = self.new_x
                        self.yc = self.new_y
                    else:
                        raise SearchStop("Inconsistent y = c")
                else:
                    self.new_x = self._map_value(
                        self.a + (1 - self.pos) * (self.b - self.a)
                    )

                    self.new_y = yield self.new_x

                    if self.new_y > self.ya + self.noise:
                        raise SearchStop("Inconsistent y > a")

                    if self.new_y < self.yb:
                        self.b, self.c = self.new_x, self.b
                        self.yb, self.yc = self.new_y, self.yb
                    elif self.new_y > self.yb:
                        self.a = self.new_x
                        self.ya = self.new_y
                    else:
                        raise SearchStop("Inconsistent y = b")
            else:
                raise SearchStop("Inconsistent a < b > c")

    def __repr__(self):
        vals = [
            (self.a, self.ya),
            (self.b, self.yb),
            (self.c, self.yc),
            (self.new_x, np.nan),
        ]
        vals.sort(key=itemgetter(0))

        format_if_not_nan = lambda x: f"{x:g}" if not np.isnan(x) else "_"

        return (
            f"Golden( "
            + " | ".join(
                f"{format_if_not_nan(x)}:{format_if_not_nan(y)}" for x, y in vals
            )
            + f" -> {min(self.ya, self.yb, self.yc):g} )"
        )


class GoldenSearcher:
    def __init__(
        self, param_name, target_precision, x0, x1, *golden_args, **golden_kwargs
    ):
        self.param_name = param_name
        self.target_precision = target_precision

        if (
            "map_value" not in golden_kwargs
            and isinstance(target_precision, int)
            and isinstance(x0, int)
            and isinstance(x1, int)
        ):
            golden_kwargs["map_value"] = int

        self.searcher = GoldenSearch(x0, x1, *golden_args, **golden_kwargs)
        self.val_gen = self.searcher.val_gen()

    def next_search_params(self, params, last_score):
        val = self.val_gen.send(last_score)

        if self.searcher.c - self.searcher.a < self.target_precision:
            raise SearchStop(f"Target precision {self.target_precision} reached")

        new_params = params.copy()
        new_params[self.param_name] = val
        return new_params

    def state_info(self):
        return str(self.searcher)

    def __repr__(self):
        return f"GoldenSearcher({self.param_name})"


class DefaultsSearcher:
    def __init__(self):
        self.num_run_left = 1

    def next_search_params(self, params, last_score):
        if self.num_run_left > 0:
            self.num_run_left -= 1
            return params

        raise SearchStop("Defaults eval finished")

    def state_info(self):
        return "Defaults"

    def __repr__(self):
        return Defaults


class ListSearcher:
    def __init__(self, param_name, val_list):
        self.param_name = param_name
        self.val_list = val_list
        self.idx = -1

    def next_search_params(self, params, last_score):
        self.idx += 1

        if self.idx == len(self.val_list):
            raise SearchStop(f"Last of {len(self.val_list)} list values reached")

        new_params = params.copy()
        new_val = self.val_list[self.idx]
        new_params[self.param_name] = new_val

        return new_params

    def state_info(self):
        return f"ListSearcher({self.param_name}, val {self.idx+1}/{len(self.val_list)})"

    def __repr__(self):
        return f"ListSearcher({self.param_name}, {len(self.val_list)} vals)"


class SearcherCV:
    def __init__(
        self,
        estimator,
        searchers,
        *,
        scoring,
        cv,
        num_feat_imps=5,
        init_best_score=None,
    ):
        self.estimator = estimator

        self.searchers = searchers

        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = init_best_score

        self.num_feat_imps = num_feat_imps

    def fit(self, X, y, verbose_search=True, **fit_params):
        if verbose_search:
            print(
                f"[{dt.datetime.now(my_timezone):%H:%M}] Starting fit on {len(X.columns)} features and {len(X)} instances with folds {self.cv} and scoring {self.scoring}"
            )
            print()

        self.best_params_ = {}

        for searcher in self.searchers:
            cur_params = self.best_params_

            if verbose_search:
                print(f">> Starting searcher {searcher}")

            try:
                score = None
                while 1:  # SearchStop expected
                    cur_params = searcher.next_search_params(
                        cur_params, score
                    )  # may throw StopSearch exception

                    mark = (
                        lambda param_name: "*"
                        if hasattr(searcher, "param_name")
                        and searcher.param_name == param_name
                        else ""
                    )

                    new_params_str = ", ".join(
                        f"{mark(param_name)}{param_name}{mark(param_name)} = {format_if_number(param_val)}"
                        for param_name, param_val in sorted(cur_params.items())
                    )

                    if verbose_search:
                        print()
                        print(
                            f"Current best score:",
                            color_score(self.best_score_)
                            if self.best_score_ is not None
                            else "-",
                        )
                        print(
                            f"Current best params:",
                            ", ".join(
                                f"{param}={format_if_number(val)}"
                                for param, val in sorted(self.best_params_.items())
                            )
                            if self.best_params_
                            else "-",
                        )
                        print(f"Searcher state: {searcher.state_info()}")
                        print(f"-> Eval: {new_params_str} .......")

                    start_time = time.time()

                    score = self._score(X, y, cur_params, fit_params)

                    end_time = time.time()
                    run_time_min = (end_time - start_time) / 60

                    new_params_color_str = ", ".join(
                        f"{color_param_name(param_name)} = {color_param_val(param_val)}"
                        if hasattr(searcher, "param_name")
                        and searcher.param_name == param_name
                        else f"{param_name} = {format_if_number(param_val)}"
                        for param_name, param_val in sorted(cur_params.items())
                    )

                    print(
                        f"....... ({run_time_min:.2g}min) {new_params_color_str} >>> {color_score(score)}"
                    )

                    if self.best_score_ is None or score < self.best_score_:  #!!!
                        self.best_score_ = score
                        self.best_params_ = cur_params.copy()

            except SearchStop as exc:
                print()
                print(f"Searcher {searcher} stopped with: {exc}")
                print()
            except Exception as exc:
                print(
                    f"Searcher {searcher} failed at params {cur_params} with: {exc}"
                )
                raise

        if verbose_search:
            print(f"Final best score: {color_score(self.best_score_)}")
            print(f"Final best params:")
            for param, val in sorted(self.best_params_.items()):
                print(f"    {color_param_name(param)} = {color_param_val(val)},")

    def _score(self, X, y, params, fit_params):
        estimator = clone(self.estimator)
        estimator.set_params(**params)

        cross_val_info = cross_validate(
            estimator,
            X,
            y,
            scoring=self.scoring,
            cv=self.cv,
            fit_params=fit_params,
            return_train_score=True,
            return_estimator=True,
        )

        for fold_idx, (clf, train_score, test_score) in enumerate(
            zip(
                cross_val_info["estimator"],
                cross_val_info["train_score"],
                cross_val_info["test_score"],
            ),
            1,
        ):
            infos = [f"{test_score:g} (train {train_score:g})"]
            if hasattr(clf, "best_iteration_") and clf.best_iteration_ is not None:
                infos.append(f"best iter {clf.best_iteration_}")

            if hasattr(clf, "best_score_") and clf.best_score_:
                best_score_str = (
                    ", ".join(
                        (f"{set_name}(" if len(clf.best_score_) > 1 else "")
                        + ", ".join(
                            f"{score_name}={score:g}"
                            for score_name, score in scores.items()
                        )
                        + (")" if len(clf.best_score_) > 1 else "")
                        for set_name, scores in clf.best_score_.items()
                    )
                    if isinstance(clf.best_score_, dict)
                    else str(clf.best_score_)
                )  # usually should always be dict
                infos.append(f"stop scores {best_score_str}")

            if hasattr(clf, "feature_importances_"):
                feat_imps = sorted(
                    zip(clf.feature_importances_, X.columns), reverse=True
                )
                infos.append(
                    "Top feat: "
                    + " · ".join(
                        str(feat) for _score, feat in feat_imps[: self.num_feat_imps]
                    )
                )

            print(f"Fold {fold_idx}:", "; ".join(infos))

        score = cross_val_info["test_score"].mean()

        return -score
