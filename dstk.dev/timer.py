import datetime


class Timer:
    def __init__(self, secondsformat="{:.1f}s", timeendformat="%H:%M:%S"):
        self.secondsformat = secondsformat
        self.timeendformat = timeendformat

        self.starttime = datetime.datetime.now()

    def duration(self):
        return datetime.datetime.now() - self.starttime

    def __str__(self):
        return str(self.duration())

    def timeend(self, ratio):
        """
        :param ratio:
        :type ratio:
        :param settime: use False if a previous __str__ call has already cached the time
        :type settime:
        """
        timenow = datetime.datetime.now()
        timedelta = timenow - self.starttime
        return (timenow + timedelta / ratio).strftime(self.timeendformat)


if __name__ == '__main__':
    import time
    t = Timer()
    time.sleep(2)
    print(t.timeend(0.01))
