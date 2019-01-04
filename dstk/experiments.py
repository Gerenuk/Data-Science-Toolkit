import logging
from typing import Any, Tuple, Dict
from dataclasses import dataclass
from .timer import Timer

_logger = logging.getLogger(__name__)


# TODO:
# no rerun when already existing
# colorlog
# product state
# persistent storage


@dataclass
class FuncCall:
    func: Any
    args: Tuple[Any]
    kwargs: Dict[str, Any]


@dataclass
class FuncCallResult:
    func_call: FuncCall
    result: Any
    duration: float


@dataclass
class FuncCallException:
    func_call: FuncCall
    exception: Any
    duration: float


class Experiments:
    def __init__(self, experiment_store=None, failed_experiment_store=None):
        self.experiment_store = experiment_store or []
        self.failed_experiment_store = failed_experiment_store or []

        self.last_logged_func_call_result = None

    def __call__(self, func, *args, **kwargs):
        func_call = FuncCall(func=func, args=args, kwargs=kwargs)

        cur_func_call_result = self.func_call_result(func_call)

        self.store_result(cur_func_call_result)
        self.log(cur_func_call_result)

        return cur_func_call_result

    @staticmethod
    def func_call_result(func_call):
        timer = Timer()

        try:
            func_result = func_call.func(*func_call.args, **func_call.kwargs)
            exception = None
        except Exception as exc:
            func_result = None
            exception = exc

        duration = timer.duration()

        if exception is None:
            return FuncCallResult(
                func_call=func_call, result=func_result, duration=duration
            )
        else:
            return FuncCallException(
                func_call=func_call, exception=exception, duration=duration
            )

    def store_result(self, func_call_result):
        if isinstance(func_call_result, FuncCallResult):
            self.experiment_store.append(func_call_result)
        else:
            self.failed_experiment_store.append(func_call_result)

    def log(self, func_call_result):
        func_name = func_call_result.func_call.func.__name__
        args_text = ", ".join(map(str, func_call_result.func_call.args))
        kwargs_text = ", ".join(
            f"{key}={val}" for key, val in func_call_result.func_call.kwargs.items()
        )
        func_call_text = (
            f"{func_name}({', '.join(filter(None, [args_text, kwargs_text]))})"
        )

        duration_text = str(func_call_result.duration)

        if isinstance(func_call_result, FuncCallResult):
            result = func_call_result.result
            _logger.info(f"{func_call_text} --> {result} (in {duration_text})")
        else:
            exception = func_call_result.exception
            _logger.error(
                f"{func_call_text} failed with {exception} (in {duration_text})"
            )


