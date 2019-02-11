from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def matches_param_names(found_param_names, required_param_names):
    return found_param_names & required_param_names and not (
        found_param_names - required_param_names
    )


class ExpSmooth:
    """
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    fitted_model / results : https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.HoltWintersResults.html
    """

    model_param_names = {
        "trend",
        "damped",
        "seasonal",
        "seasonal_periods",
        "dates",
        "freq",
        "missing",
    }

    def __init__(self, **params):
        """
        params:
        * from init: https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
        * from fit: https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.fit.html
        """
        self.params = params
        self.raw_model = None
        self._model = None

    def fit(self, Xt):
        """
        Xt should have time period for index
        """
        self.raw_model = ExponentialSmoothing(
            Xt, **{k: v for k, v in self.params.items() if k in self.model_param_names}
        )

        self._model = self.raw_model.fit(
            **{k: v for k, v in self.params.items() if k not in self.model_param_names}
        )

    @property
    def model(self):
        if self._model is None:
            raise ValueError("Model has to be fitted first with .fit()")

        return self._model

    def predict(self, **kwargs):
        """
        kwargs:
        * from predict: https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.HoltWintersResults.predict.html
        * from forecast: https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.HoltWintersResults.forecast.html
        """
        if matches_param_names(kwargs, {"steps"}):
            return self.model.forecast(**kwargs)

        return self.model.predict(**kwargs)

    @property
    def aic(self):
        return self.model.aic


class ARIMA:
    """
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    fitted_model / results : https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html
    """

    model_param_names = {
        "endog",
        "exog",
        "order",
        "seasonal_order",
        "trend",
        "measurement_error",
        "time_varying_regression",
        "mle_regression",
        "simple_differencing",
        "enforce_stationarity",
        "enforce_invertibility",
        "hamilton_representation",
        "concentrate_scale",
    }

    def __init__(self, **params):
        """
        params:
        * from init: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
        * from fit: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html
        """
        self.params = params
        self.raw_model = None
        self._model = None

    def fit(self, Xt):
        self.raw_model = SARIMAX(
            Xt, **{k: v for k, v in self.params.items() if k in self.model_param_names}
        )

        self._model = self.raw_model.fit(
            **{k: v for k, v in self.params.items() if k not in self.model_param_names}
        )

    @property
    def model(self):
        if self._model is None:
            raise ValueError("Model has to be fitted first with .fit()")

        return self._model

    def predict(self, **kwargs):
        """
        kwargs:
        * from get_prediction: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_prediction.html
        * from forecast: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.forecast.html
        """
        if matches_param_names(kwargs, {"steps"}):
            return self.model.forecast(**kwargs)

        return self.model.get_prediction(**kwargs)

    @property
    def aic(self):
        return self.model.aic
