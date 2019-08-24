from collections import namedtuple
from math import sqrt
from typing import NamedTuple, Any
from sklearn.base import clone
from sklearn.model_selection import cross_validate
import numpy as np
from operator import itemgetter
import time


class GoldenInconsistent(Exception):
    def __init__(self, golden_state):
        self.golden_state = golden_state


class GoldenState(NamedTuple):
    a: Any
    b: Any
    c: Any
    new_x: Any
    ya: Any
    yb: Any
    yc: Any

    def __repr__(self):
        vals = [
            (self.a, self.ya),
            (self.b, self.yb),
            (self.c, self.yc),
            (self.new_x, np.nan),
        ]
        vals.sort(key=itemgetter(0))
        return f"GoldenState( " + " | ".join(f"{x:g}:{y:g}" for x, y in vals) + " )"


def golden_minimize(
    xs, ys=None, min_bound=True, max_bound=True, map_value=None, noise=0
):
    """
    Pass 2 or 3 initial values
    Range may extend within bounds if minimum appears at edge

    Example:
    def func(x):
        return x**2

    xs = [-10, 5]
    mini = golden_minimize(xs, [func(x) for x in xs])
    x, state = next(mini)
    for _ in range(10):
        x, state = mini.send(func(x))

    The case a b b a  with b < a is not solved at throws GoldenInconsistent
    """
    phi = (1 + sqrt(5)) / 2
    pos = 2 - phi  # ~0.382

    if ys is None:
        ys = []
        for x in xs:
            y = yield x, None
            ys.append(y)

    if len(xs) == 2:
        a, c = xs
        ya, yc = ys
        b = a + pos * (c - a)
        if map_value is not None:
            b = map_value(b)

        yb = yield b, GoldenState(a, b, c, np.nan, ya, np.nan, yc)
    else:
        a, b, c = xs
        ya, yb, yc = ys

    if min_bound is True:
        min_bound = a

    if max_bound is True:
        max_bound = c

    new_x = np.nan
    new_y = np.nan

    while 1:
        # print(">>>", a, b, c)
        assert a < b < c, "x coordinates not ordered"

        d1 = b - a
        d2 = c - b

        if ya < yb <= yc + noise:  # extend region left
            if min_bound < a:
                new_x = b - d1 / pos
                if map_value is not None:
                    new_x = map_value(new_x)

                if new_x < min_bound:
                    new_x = min_bound

                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc)
                a, b, c = new_x, a, b
                ya, yb, yc = new_y, ya, yb
            else:
                new_x = a + pos * (b - a)
                if map_value is not None:
                    new_x = map_value(new_x)

                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc)
                b, c = new_x, b
                yb, yc = new_y, yb

        elif ya + noise >= yb > yc:  # extend region right
            if c < max_bound:
                new_x = b + d2 / pos
                if map_value is not None:
                    new_x = map_value(new_x)

                if new_x > max_bound:
                    new_x = max_bound

                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc)
                a, b, c = b, c, new_x
                ya, yb, yc = yb, yc, new_y
            else:
                new_x = c - pos * (c - b)
                if map_value is not None:
                    new_x = map_value(new_x)

                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc)
                a, b = b, new_x
                ya, yb = yb, new_y

        elif ya >= yb - noise and yb - noise <= yc:
            if d1 < d2:
                new_x = c - (1 - pos) * (c - b)
                if map_value is not None:
                    new_x = map_value(new_x)

                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc)

                if new_y > yc + noise:
                    raise GoldenInconsistent(GoldenState(a, b, c, new_x, ya, yb, yc))

                if new_y < yb:
                    a, b = b, new_x
                    ya, yb = yb, new_y
                elif new_y > yb:
                    c = new_x
                    yc = new_y
                else:
                    raise GoldenInconsistent(GoldenState(a, b, c, new_x, ya, yb, yc))
            else:
                new_x = a + (1 - pos) * (b - a)
                if map_value is not None:
                    new_x = map_value(new_x)

                new_y = yield new_x, GoldenState(a, b, c, new_x, ya, yb, yc)

                if new_y > ya + noise:
                    raise GoldenInconsistent(GoldenState(a, b, c, new_x, ya, yb, yc))

                if new_y < yb:
                    b, c = new_x, b
                    yb, yc = new_y, yb
                elif new_y > yb:
                    a = new_x
                    ya = new_y
                else:
                    raise GoldenInconsistent(GoldenState(a, b, c, new_x, ya, yb, yc))
        else:
            raise GoldenInconsistent(GoldenState(a, b, c, new_x, ya, yb, yc))


class BoundTracer:
    def __init__(self, param_name, min_bound, max_bound, max_diff, noise=0):
        self.param_name = param_name
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.max_diff = max_diff
        self.noise = noise

    def create_minimizer(self, params):
        if isinstance(self.min_bound, int) and isinstance(self.max_bound, int):
            map_value = int
        else:
            map_value = None

        return golden_minimize(
            [self.min_bound, self.max_bound], map_value=map_value, noise=self.noise
        )

    def if_continue_with(self, state):
        return state is None or (state.c - state.a) > self.max_diff


class GoldenSearchCV:
    def __init__(self, estimator, params_trace, scoring, cv):
        self.estimator = estimator

        self.params_trace = params_trace

        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, X, y, verbose_search=False, **fit_params):
        cur_params = {}
        for tracer in self.params_trace:
            if isinstance(tracer, tuple):
                tracer = BoundTracer(*tracer)

            param_name = tracer.param_name
            minimizer = tracer.create_minimizer(cur_params)
            param_val, state = next(minimizer)
            try:
                while tracer.if_continue_with(state):
                    if verbose_search:
                        print(f"Golden state: {state}")
                        print(f"Eval: {param_name} = {param_val:g} ...")

                    cur_params[param_name] = param_val

                    start_time = time.time()
                    score = self._score(X, y, cur_params, fit_params)
                    end_time = time.time()

                    print(
                        f"... ({(end_time-start_time)/60:.2g}min) {param_name} = {param_val:g} >>> {score:g}"
                    )

                    if self.best_score_ is None or score < self.best_score_:  #!!!
                        self.best_score_ = -score
                        self.best_params_ = cur_params.copy()

                    param_val, state = minimizer.send(score)
            except GoldenInconsistent as exc:
                print(f"Terminating on {exc}")
            else:
                print(f"Search finished by the tracer")

    def _score(self, X, y, params, fit_params):
        estimator = clone(self.estimator)
        estimator.set_params(**params)

        score = cross_validate(
            estimator, X, y, scoring=self.scoring, cv=self.cv, fit_params=fit_params
        )["test_score"].mean()

        return -score
