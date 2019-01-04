import time


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return time.time() - self.start_time

    def __str__(self, dynamic_units=True):
        duration = self.duration()

        if dynamic_units:
            if duration < 1:
                duration_text = f"{duration*1000:.0f}msec"
            elif duration < 60:
                duration_text = f"{duration:.1f}sec"
            else:
                duration_text = f"{duration/60:.1f}min"
        else:
            duration_text = f"{duration:g}sec"

        return duration_text


def estimated_end_time(timer: Timer, progress: float, time_end_format="%H:%M:%S"):
    time_now = time.time()
    time_delta = time_now - timer.start_time
    time_end = time_now + time_delta / progress
    time_text = time.strftime(time_end_format, time.localtime(time_end))
    return time_text
