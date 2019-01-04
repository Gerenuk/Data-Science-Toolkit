from statsmodels.tsa.holtwinters import ExponentialSmoothing


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
        self.params = params
        self.raw_model = None
        self.fitted_model = None

    def fit(self, Xt):
        """
        Xt should have time period for index
        """
        self.raw_model = ExponentialSmoothing(
            Xt, **{k: v for k, v in self.params.items() if k in self.model_param_names}
        )

        self.fitted_model = self.raw_model.fit(
            **{k: v for k, v in self.params.items() if k not in self.model_param_names}
        )

    @property
    def model(self):
        if self.fitted_model is None:
            raise ValueError("Model has to be fitted first with .fit()")

        return self.fitted_model

    def predict(self, num_forecast):
        return self.model.forecast(num_forecast)

    @property
    def score(self):
        return self.model.aic
