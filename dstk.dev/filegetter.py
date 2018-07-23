import os.path
import datetime


class FileGetter:
    def __init__(self, *dirs):
        self.dir = os.path.join(*[str(d) for d in dirs])

    def __call__(self, *dirs):
        return os.path.join(self.dir, *[str(d) for d in dirs])

    def open(self, *dirs, options="r", **kwargs):
        return open(self(*dirs), options, **kwargs)

    def timestamp_filename(self, filename, timeformat="%m.%d %Hh%Mm%S"):
        # for format see: http://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
        timenow = datetime.datetime.now()
        filename = filename.format(timenow.strftime(timeformat))
        filename = os.path.join(self.dir, filename)
        if os.path.exists(filename):
            count = 1
            while 1:
                ext_filename = filename + "-{}".format(count)
                if not os.path.exists(ext_filename):
                    return ext_filename
                count += 1
        return filename

    def open_timestamp_filename(self, filename, options="w", timeformat="%m.%d %Hh%Mm%S", **kwargs):
        return open(self.timestamp_filename(filename, timeformat), options, **kwargs)
