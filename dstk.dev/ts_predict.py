import warnings
from abc import abstractmethod
from typing import Any

import fbprophet
import pandas as pd
import statsmodels as sm
from dataclasses import dataclass


# TODO:
# include auto arima
# plot conf interval
# can I still log warnings if I catch them?
# check actual warnings; maybe fix
# fit_predict_multiple: squeeze if length=1?


@dataclass
class TimeseriesPrediction:
    pred: Any
    real: Any
    name: str = ""

    def __post_init__(self):
        if self.name != "":
            self.pred = self.pred.rename("pred_" + self.name)
            self.real = self.real.rename(self.name)

    def meanabs(self):
        return sm.tools.eval_measures.meanabs(self.real, self.pred)

    def plot(self):
        self.real.plot()
        self.pred.plot()
        plt.grid(True)
        plt.legend()


class TimeseriesModel:
    @abstractmethod
    def fit(self, ts, **fit_params):
        pass

    @abstractmethod
    def predict(self, pred_start, pred_end, **pred_params):
        """
        returns predicted timeseries with date index
        """
        pass

    def fit_predict(self, ts, pred_start, pred_end, fit_params=None, pred_params=None):
        """
        Will limit training to the period before `pred_start`
        """
        fit_params = fit_params or {}
        pred_params = pred_params or {}

        self.fit(ts[:pred_start].iloc[:-1], **fit_params)

        ts_pred = self.predict(pred_start, pred_end, **pred_params)
        ts_real = ts[pred_start:pred_end]

        return TimeseriesPrediction(ts_pred, ts_real)

    def fit_predict_multiple(
        self,
        ts,
        pred_start,
        pred_end,
        step=1,
        length=None,
        fit_params=None,
        pred_params=None,
    ):
        if length is None:
            length = step
            
        length-=1   # since first point also counts

        dates = ts[pred_start:pred_end].index

        results = []
        for i in range(0, len(dates) - length, step):
            iter_pred_start = dates[i]
            iter_pred_end = dates[i + length]
            pred = self.fit_predict(
                ts,
                iter_pred_start,
                iter_pred_end,
                fit_params=fit_params,
                pred_params=pred_params,
            )
            results.append(pred)

        return results


class ARIMA(TimeseriesModel):
    def __init__(self, **model_params):
        """
        model_params: http://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
        """
        self.model_params = model_params

        self.model_ = None
        self.fit_result_ = None
        self.pred_result_ = None

    def fit(self, ts, show_warnings=False, **fit_params):
        """
        ts : should have date index
        fit_params: http://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html
        """
        with warnings.catch_warnings():
            if not show_warnings:
                warnings.simplefilter("ignore")

            self.model_ = sm.tsa.statespace.sarimax.SARIMAX(ts, **self.model_params)
            self.fit_result_ = self.model_.fit(**fit_params)

    def predict(self, pred_start, pred_end, show_warnings=False, **pred_params):
        """
        pred_params: http://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.get_prediction.html
        """
        if "dynamic" not in pred_params:
            pred_params["dynamic"] = True

        with warnings.catch_warnings():
            if not show_warnings:
                warnings.simplefilter("ignore")
            pred_result = self.fit_result_.get_prediction(
                start=pred_start, end=pred_end, **pred_params
            )
            self.pred_result_ = pred_result

            ts_pred = pred_result.predicted_mean

        return ts_pred


class ExpSmoothing(TimeseriesModel):
    def __init__(self, **model_params):
        """
        model_params: http://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html#statsmodels.tsa.holtwinters.ExponentialSmoothing
        self.fit_result_: http://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.HoltWintersResults.html
        """
        for name, val in [("trend", "mul"), ("damped", True)]:
            if name not in model_params:
                model_params[name] = val

        self.model_params = model_params

        self.model_ = None
        self.fit_result_ = None
        self.pred_result_ = None

    def fit(self, ts, show_warnings=False, **fit_params):
        """
        ts : should have date index
        fit_params: http://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html
        """
        with warnings.catch_warnings():
            if not show_warnings:
                warnings.simplefilter("ignore")

            self.model_ = sm.tsa.holtwinters.ExponentialSmoothing(
                ts, **self.model_params
            )
            self.fit_result_ = self.model_.fit(**fit_params)

    def predict(self, pred_start, pred_end, show_warnings=False):
        """
        pred_params: http://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.get_prediction.html
        """
        with warnings.catch_warnings():
            if not show_warnings:
                warnings.simplefilter("ignore")
            self.pred_result_ = self.fit_result_.predict(start=pred_start, end=pred_end)
            ts_pred = self.pred_result_

        return ts_pred


class Prophet(TimeseriesModel):
    def __init__(self, **model_params):
        """
        model_params: https://github.com/facebook/prophet/blob/master/python/fbprophet/forecaster.py#L45
        """
        self.model_params = model_params

        self.model_ = None
        self.pred_result_ = None
        self._inferred_freq = None  # needed for prediction

    def fit(self, ts, show_warnings=False, **fit_params):
        """
        fit_params: https://github.com/facebook/prophet/blob/master/python/fbprophet/forecaster.py#L900
        """
        self.model_ = fbprophet.Prophet(**self.model_params)

        with warnings.catch_warnings():
            if not show_warnings:
                warnings.simplefilter("ignore")

            self._inferred_freq = ts.index.inferred_freq

            df_train = pd.DataFrame({"ds": ts.index, "y": ts})
            self.model_.fit(df_train, **fit_params)

    def predict(self, pred_start, pred_end, show_warnings=False):
        with warnings.catch_warnings():
            if not show_warnings:
                warnings.simplefilter("ignore")

            df_pred = self.model_.predict(
                pd.DataFrame(
                    {
                        "ds": pd.date_range(
                            pred_start, pred_end, freq=self._inferred_freq
                        )
                    }
                )
            )  # freq not set here

            self.pred_result_ = df_pred
            ts_pred = df_pred.set_index("ds")["yhat"]

        return ts_pred
