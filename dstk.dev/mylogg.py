import inspect
import logging
import traceback
import sys
import os
import bz2
import datetime
import time

"""
IDEAS:
* store avg time in db
* predict runtime
* need self.logger or just re-direct from outside?
* dont track average on exception
"""

std_logger = logging.getLogger(__name__)


class ExcSafe:
    def __init__(self, catch_exc_type=Exception, logger_func=std_logger.warning, name=None, stop_exception=True, show_trace_level_num=1):
        self.catch_exc_type = catch_exc_type
        self.logger_func = logger_func
        self.name = name
        self.stop_exception=stop_exception
        self.show_trace_level_num=show_trace_level_num

    def __call__(self, func):
        def exc_save_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except self.catch_exc_type as exc_val:
                exc_type, _exc_value, exc_traceback = sys.exc_info()
                filename, fileline, _func, text = traceback.extract_tb(exc_traceback)[self.show_trace_level_num]
                self.logger_func(
                    "Skipping rest of {} '{}' which failed in file '{}'[{}] at source line '{}' due to exception: {} {}".format(
                        type(func).__name__,
                        func.__name__,
                        os.path.basename(filename),
                        fileline,
                        text,
                        exc_type.__name__,
                        exc_val))

        return exc_save_func

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        filename, fileline, _func, text = traceback.extract_tb(exc_tb, 1)[0]
        self.logger_func(
            "Skipping context {}which failed in file '{}'[{}] at source line '{}' due to exception: {} {}".format(
                self.name + " " if self.name is not None else "",
                os.path.basename(filename),
                fileline,
                text,
                exc_type.__name__,
                exc_val))

        return self.stop_exception


@ExcSafe()
def source(obj):
    filename = inspect.getsourcefile(obj)
    sourcelines, startline = inspect.getsourcelines(obj)

    return "".join(["{} '{}' defined in {}[{}]\n".format(type(obj).__name__,
                                                         getattr(obj, "__name__", "??"),
                                                         filename,
                                                         startline)] + sourcelines)


@ExcSafe()
def params(names, multiline=True, format=None):
    name_list = names.split()

    if format is None:
        format = ["!r"] * len(name_list)
    elif isinstance(format, str):
        format = [format] * len(name_list)
    if len(format) < len(name_list):
        format.extend(["!r"] * (len(name_list) - len(format)))

    global_params = globals()
    param_result = []
    for name, format in zip(name_list, format):
        if name in global_params:
            param_result.append(("{} = {" + format + "}").format(name, global_params[name]))
        else:
            param_result.append("{} is N/A".format(name))

    if multiline:
        return "Params:\n" + "\n".join(param_result)
    else:
        return "Params: " + ", ".join(param_result)


def save_file(filename, dest_dir=None, logger_func=std_logger.info):
    """
    :param filename: use __file__ for current filename
    :param dest_dir:
    :param logger:
    :return: None
    """
    if dest_dir is None:
        dest_dir = os.path.dirname(filename)

    time_tag = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")

    base_filename = "{} {}.bz2".format(time_tag, os.path.basename(filename))
    dest_filename = os.path.join(dest_dir, base_filename)

    with bz2.BZ2File(dest_filename, "w") as dest_file, open(filename, "rb") as src_file:
        for line in src_file:
            dest_file.write(line)

    logger_func("Saved file '{}' as '{}'".format(filename, dest_filename))


class LogCall:
    def __init__(self, logger_func=std_logger.info, log_start=False):
        self.logger_func = logger_func
        self.log_start = log_start

    def __call__(self, func):
        def log_call_func(*args, **kwargs):
            func_call_text = "{}({}{})".format(func.__name__,
                                               ", ".join(repr(arg) for arg in args),
                                               "" if not kwargs else ", ".join(
                                                   "{} = {!r}".format(key, val) for key, val in kwargs.items())
            )
            if self.log_start:
                self.logger_func("Calling {}".format(func_call_text))

            result = func(*args, **kwargs)

            self.logger_func("Return value >> {!r} << from call {}".format(result, func_call_text))
            return result

        return log_call_func


def std_timer_formatter(total_seconds):
    hours = total_seconds // 3600
    minutes = total_seconds // 60 % 60
    seconds = total_seconds % 60
    return ("{:02}h".format(hours) if hours else "" +
                                                 "{:02}m".format(minutes) if hours else "" +
                                                                                        "{:02.0f}s".format(seconds))


class Timer:
    """
    ATM only __exit__ keeps track of self.total_durations
    """

    def __init__(self, name=None, logger_func=std_logger.info, formatter=std_timer_formatter, func_arg_num_iter=None,
                 log_predict=True):
        self.name = name
        self.logger_func = logger_func
        self.formatter = formatter
        self.num_iter = 1
        self.func_arg_num_iter = func_arg_num_iter
        self.log_predict = log_predict
        self.start_time = None
        self.total_durations = []
        self.total_iterations = []

    def start(self):
        self.start_time = time.time()

    def __enter__(self):
        """
        Need to set Timer.num_iter during with block
        :return:
        """
        self.start()
        if self.log_predict and self.total_durations:
            avg_job_duration = self.avg_duration() * self.num_iter
            estimated_finish_time = self.start_time + avg_job_duration
            self.logger_func("Starting job {}with estimated running time {:.1f}s {}and finish time {}".format(
                "'{}' ".format(self.name) if self.name else "",
                avg_job_duration,
                "for {} iterations ".format(self.num_iter) if self.num_iter > 1 else "",
                datetime.datetime.fromtimestamp(estimated_finish_time).strftime("%H:%M:%S")))

    def duration(self):
        return time.time() - self.start_time

    def avg_duration(self):
        return sum(self.total_durations) / sum(self.total_iterations)

    def __call__(self, func):
        """
        Function decorator to always log time
        :param func:
        :return:
        """

        def timed_func(*args, **kwargs):
            with self:
                self.num_iter = self.func_arg_num_iter(*args, **kwargs) if self.func_arg_num_iter else 1
                func(*args, **kwargs)

        self.name = "func " + func.__name__
        return timed_func

    def __str__(self):
        return self.formatter(self.duration())

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.duration()
        self.total_durations.append(duration)
        self.total_iterations.append(self.num_iter)
        output = []
        output.append("Job {}finished after {}{}".format(
            "'{}' ".format(self.name) if self.name is not None else "",
            str(self),
            " with EXCEPTION" if exc_type is not None else ""))
        if len(self.total_durations) > 1:
            output.append("{:.1f}s per call ({} calls)".format(sum(self.total_durations) / len(self.total_durations),
                                                               len(self.total_durations)))
        if self.num_iter > 1:
            output.append("{:.1f}s per all {} iterations".format(duration / self.num_iter, self.num_iter))
        if len(self.total_durations) > 1 and any(x > 1 for x in self.total_iterations):
            output.append("{:.1f}s per iteration overall ({} iterations)".format(
                self.avg_duration(), sum(self.total_iterations)))
        self.logger_func("; ".join(output))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    @LogCall()
    def test(a, b):
        # print(a)
        return a

    class A:
        pass

    @ExcSafe()
    def exc_test(a):
        a = {}
        a[1]


    # print(source(A))
    a = 14
    b = 13
    # print(params("a b c", multiline=False))

    # exc_test(2)

    # with ExcSafe(name="myBlock"):
    # raise Exception("ABC")
    # save_file(r"F:\V\VT_Oberursel\RBV\KA\CRM_DM_Mafo\04_CMI\03_Projekte\Python\PyCharm\mystd_logger.py")

    #test(1, 2)
    #test("b", 4)
    #with Timer("mytimer", 10):
    #    time.sleep(2)
    #print(t)

    @Timer(func_arg_num_iter=lambda *args, **kwargs: args[0])
    @ExcSafe()
    def timetest(iter_, time_=1):
        time.sleep(time_)
        raise Exception("ABC")

    timetest(10, 1)
    #timetest(20, 2)
    #timetest(30, 3)