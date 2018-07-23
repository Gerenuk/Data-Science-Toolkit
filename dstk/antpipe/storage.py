"""
Storage used for data sources in the dependency framework:
* .read(): read data
* .write(data): write data
* .timestamp: return timestamp compatible with other storages
"""

import os
import logging


logger = logging.getLogger(__name__)


def hdfs_timestamp(filename):
    import subprocess
    import dateutil
    import os

    upper_dir = os.path.abspath(os.path.join(filename, ".."))
    out = subprocess.getoutput('hdfs dfs -ls "{}"'.format(upper_dir))
    out = [line.split() for line in out.split("\n")[1:]]

    file_out = [line for line in out if line[7] == filename]

    assert len(file_out) <= 1
    if len(file_out) == 0:
        raise FileNotFoundError("hdfs:/" + filename)

    time = dateutil.parser.parse("{}T{}".format(file_out[0][5], file_out[0][6]))

    return time._timestamp()


class ParquetStorage:
    def __init__(self, filename, spark):
        self.filename = filename
        self.spark = spark
        self.name = "File hdfs:/{}".format(self.filename)

    def read(self):
        return self.spark.read.parquet(self.filename)

    def write(self, data):
        data.write.parquet(self.filename, mode="overwrite")

    @property
    def timestamp(self):
        return hdfs_timestamp(self.filename)


class ParquetStorageReadOnlyKeepTimestamp(ParquetStorage):
    def __init__(self, spark, filename):
        self.filename = filename
        self.spark = spark
        self.timestamp_ = None

    def write(self, data):
        logger.info("Skipping write operation for {}".format(self.filename))

    @property
    def timestamp(self):
        if self.timestamp_ is None:
            self.timestamp_ = hdfs_timestamp(self.filename)
        return self.timestamp_


class ParquetFilename:
    def __init__(self, filename):
        self.filename = filename
        self.name = "File hdfs:/{}".format(self.filename)

    def read(self):
        return self.filename

    def write(self, data):
        raise ValueError("ParquetFilename for file '{}' is not intended for writing".format(self.filename))

    @property
    def timestamp(self):
        return hdfs_timestamp(self.filename)


class ParquetStorageReadOnly(ParquetStorage):
    def write(self, data):
        logger.info("Skipping write operation for {}".format(self.filename))


class FileStorage:
    def __init__(self, filename):
        self.filename = filename
        self.name = filename

    def read(self):
        return open(self.filename).read()

    def write(self, data):
        with open(self.filename, "w") as f:
            f.write(data)

    @property
    def timestamp(self):
        if not os.path.exists(self.filename):
            return 0  # early timestamp

        return os.path.getmtime(self.filename)


class MemStorage:
    def __init__(self, data=None, timestamp=0, name="MEM"):
        self.data = data
        self.timestamp = timestamp
        self.name = name

    def read(self):
        return self.data

    def write(self, data):
        self.data = data

