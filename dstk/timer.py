import time
from dataclasses import dataclass


@dataclass
class TimerDuration:
    value: float

    def __str__(self):
        duration_seconds = self.value

        if duration_seconds < 1:
            duration_text = f"{duration_seconds*1000:.0f}msec"
        elif duration_seconds < 60:
            duration_text = f"{duration_seconds:.1f}sec"
        else:
            duration_text = f"{duration_seconds/60:.1f}min"

        return duration_text


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return TimerDuration(value=time.time() - self.start_time)

    def __repr__(self):
        return f"Timer()"  # repr in debugging with time not meaningful

    def __str__(self):
        duration = self.duration()
        return f"Timer({duration})"

    @staticmethod
    def _time_text(time_, time_format="%H:%M:%S"):
        return time.strftime(time_format, time.localtime(time_))

    def estimated_end_time(self, cur_progress: float):
        time_now = time.time()
        time_delta = time_now - self.start_time
        time_end = time_now + time_delta / cur_progress

        time_text = self._time_text(time_end)
        return time_text
