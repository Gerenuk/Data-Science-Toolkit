import os.path
import datetime
import string


class FilePath:
    def __init__(self, *paths):
        self.paths = paths

    def add_path(self, *paths):
        return FilePath(self.paths + paths)

    def __call__(self, *paths, on_exists="ignore"):
        """
        if_exists: ignore=just return the name; error=exception if file exists; rename=give alternative name if file exists
        """
        filepath = os.path.join(*(self.paths + paths))
        if on_exists == "ignore":
            return filepath

        file_exists = os.path.exists(filepath)

        if on_exists == "error" and file_exists:
            raise ValueError("Filename {} already exists".format(filepath))

        if on_exists == "rename":
            count = 1
            filebase, fileext = os.path.splitext(filepath)
            while file_exists:
                filepath = "{}-{}{}".format(filebase, count, fileext)
                count += 1
                file_exists = os.path.exists(filepath)
            return filepath

        raise ValueError("Unknown on_exists parameter {}".format(on_exists))

    def timestamp_filename(self, filename, timeformat="%m.%d %Hh%Mm%S"):
        # for format see: http://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
        time_text = datetime.datetime.now().strftime(timeformat)
        filename = string.Template(filename).safe_substitute(time=time_text)

        filepath = self.__call__(filename)
        return filepath
