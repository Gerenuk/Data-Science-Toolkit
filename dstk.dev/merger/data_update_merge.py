"""
* The filenames need to be sorted in chronological/alphabetical order since only the last file (in EACH glob pattern) we be taken as
  an incomplete file (e.g. only half month of data) and be replaced on a new update) -> only the last file in EACH glob pattern can
  be partial
* Filename in cache_filelist.txt have their data already transformed into the cache.txt file (they are NOT being re-read if they change)
* For a full re-read delete both cache*.txt files
* The output header needs to be specified for CsvDest

TODO IDEAS:
* save cache backup
* sentinel field what's already in there
* validate input first
* error code at the end
* show percent error
* show progress how many byte percent already read
"""

import os
import csv
import glob
import logging
import re

from helpers import copy_file


logger = logging.getLogger(__name__)


def file_line_number(file):
    return sum(1 for _line in file)


class CsvSrc:
    """
    stores filename
    has .data_id and .is_final
    """

    def __init__(
        self, filename, is_final, csv_format, file_format=None, use_dictreader=True
    ):
        self.is_final = is_final
        self.data_id = os.path.basename(filename)
        self.filename = filename
        self.csv_format = csv_format
        self.file_format = (
            file_format
            if file_format is not None
            else dict(newline="", encoding="utf8")
        )
        self.use_dictreader = use_dictreader

    def __str__(self):
        return "CsvSrc({})".format(self.filename)

    def __iter__(self):
        with open(self.filename, **self.file_format) as file:
            if self.use_dictreader:
                reader = csv.DictReader(file, **self.csv_format)
            else:
                reader = csv.reader(file, **self.csv_format)
            for row in reader:
                yield row

    def __len__(self):
        return file_line_number(open(self.filename, **self.file_format)) - 1


class CsvCacheList:
    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(self.filename):
            self.cache_content = set(
                line[:-1] for line in open(self.filename, encoding="utf8")
            )
        else:
            self.cache_content = set()
        self.cache_filelist_writer = open(self.filename, "a+", encoding="utf8")

    def __contains__(self, id_):
        return id_ in self.cache_content

    def id_set(self):
        return self.cache_content

    def add(self, id_):
        assert id_ not in self.cache_content
        self.cache_filelist_writer.write(id_ + "\n")
        self.cache_content.add(id_)


class CacheWriters:
    """
    Combines multiple writers
    Adds checking whether something already in cache
    """

    def __init__(self, cache_filelist, *dest_L):
        """
        dest_L: need .writer(), .finaL_writer(), .close()
        """
        self.cache_filelist = cache_filelist
        self.dest_L = dest_L

    def cache_id_set(self):
        return self.cache_filelist.id_set()

    def writers(self, data_id, is_final):
        if is_final:
            if data_id not in self.cache_filelist:
                self.cache_filelist.add(data_id)
                writer_L = [dest.final_writer() for dest in self.dest_L]
                logging.debug(
                    "Final source {} written to cache since it is not in cache".format(
                        data_id
                    )
                )
            else:
                logging.debug(
                    "Not writing final source {} since it is already in cache".format(
                        data_id
                    )
                )
                return None
        else:
            writer_L = [dest.writer() for dest in self.dest_L]
            logging.debug("Part source {} written".format(data_id))

        return writer_L

    def close(self):
        for dest in self.dest_L:
            dest.close()


class CsvDestWriter:
    """
    Stores the filenames of destination and cache
    Keeps data consistent
    Returns appropriate writers (having .writerow()) on .writer() or final_writer()
    """

    def __init__(
        self,
        dest_filename,
        header,
        csv_format=None,
        cache_filename=None,
        file_format=None,
        cache_backup=None,
        use_dictwriter=True,
    ):
        self.dest_filename = dest_filename
        self.cache_filename = (
            cache_filename
            if cache_filename is not None
            else "{}_CACHE{}".format(*os.path.splitext(self.dest_filename))
        )
        self.header = header
        self.file_format = (
            file_format
            if file_format is not None
            else dict(newline="", encoding="utf8")
        )
        self.csv_format = csv_format if csv_format is not None else dict(delimiter=";")
        if cache_backup is not None:
            root, ext = os.path.splitext(self.cache_filename)
            self.cache_backup_filename = "{}_BACKUP{}".format(root, ext)
        else:
            self.cache_backup_filename = None

        self.use_dictwriter = use_dictwriter

        self.dest_file = None
        self.dest_writer = None
        self.cache_file = None
        self.cache_writer = None
        self.cache_copied = False

    def cache_id_set(self):
        return self.cache_filelist.id_set()

    def __str__(self):
        return "CsvDest({}; cache {})".format(self.dest_filename, self.cache_filename)

    def final_writer(self):
        if self.cache_copied:
            raise ValueError(
                "Cannot write_final_row() after the cache has already been copied (due to write_row()). Put all write_row() at the end."
            )

        assert self.dest_writer is None

        if self.cache_writer is None:
            if self.cache_backup_filename is not None and os.path.exists(
                self.cache_filename
            ):
                logging.debug(
                    "Making cache backup {}".format(self.cache_backup_filename)
                )
                copy_file(self.cache_filename, self.cache_backup_filename)
            self.cache_file, self.cache_writer = self._open_file_with_header(
                self.cache_filename
            )

        return self.cache_writer

    def writer(self):
        assert self.cache_writer is None or self.dest_writer is None

        if self.dest_writer is None:
            if os.path.exists(self.cache_filename):
                self._copy_cache()
            self.dest_file, self.dest_writer = self._open_file_with_header(
                self.dest_filename
            )

        assert self.cache_writer is None

        return self.dest_writer

    def _open_file_with_header(self, filename):
        if os.path.exists(filename):
            file = open(filename, "a", **self.file_format)
            if self.use_dictwriter:
                writer = csv.DictWriter(file, self.header, **self.csv_format)
            else:
                writer = csv.writer(file, **self.csv_format)
            logging.debug("File {} opened for appending".format(filename))
        else:
            file = open(filename, "w", **self.file_format)
            if self.use_dictwriter:
                writer = csv.DictWriter(file, self.header, **self.csv_format)
                writer.writeheader()
            else:
                writer = csv.writer(file, **self.csv_format)
                writer.writerow(self.header)
            logging.debug("File {} created and header written".format(filename))
        return file, writer

    def _copy_cache(self):
        logging.debug("Closing and copying cache")
        if self.cache_writer is not None:
            self.cache_writer = None
            self.cache_file.close()
        copy_file(self.cache_filename, self.dest_filename)
        self.cache_copied = True

    def close(self):
        logging.debug("Closing CsvDest")
        if self.cache_writer is not None and self.dest_writer is None:
            self._copy_cache()

        if self.dest_writer is not None:
            self.dest_writer = None
            self.dest_file.close()


def glob_csv_sources(
    final_glob_filename, part_glob_filename, row_transform, csv_format
):  # currently unused
    result = []
    for filename in glob.glob(final_glob_filename):
        result.append(CsvSrc(filename, row_transform, True, csv_format))
    for filename in glob.glob(part_glob_filename):
        result.append(CsvSrc(filename, row_transform, False, csv_format))
    logging.debug("Created CSV sources with {} elements".format(len(result)))
    return result


def csv_source_part_last(filename_IL, csv_format, use_dictreader=True):
    result = []
    result_part = []
    for filename_L in filename_IL:
        logging.debug("Search files {}".format(filename_L))
        filename_sorted_L = sorted(filename_L)
        if not filename_sorted_L:
            logging.warning("No file found for {}".format(filename_L))
            continue
        for filename in filename_sorted_L[:-1]:
            logging.debug("Adding {} as final".format(filename))
            result.append(
                CsvSrc(filename, True, csv_format, use_dictreader=use_dictreader)
            )
        logging.debug("Adding {} as part".format(filename_sorted_L[-1]))
        result_part.append(
            CsvSrc(
                filename_sorted_L[-1], False, csv_format, use_dictreader=use_dictreader
            )
        )
    return result + result_part


def csv_source_part_regex(
    filename_L, regex, csv_format, file_format=None, use_dictreader=True
):
    result = []
    result_part = []
    for filename in filename_L:
        m = re.search(regex, filename)
        if m:
            logging.debug("Adding {} as part".format(filename))
            result_part.append(
                CsvSrc(
                    filename,
                    False,
                    csv_format,
                    file_format=file_format,
                    use_dictreader=use_dictreader,
                )
            )
        else:
            logging.debug("Adding {} as final".format(filename))
            result.append(
                CsvSrc(
                    filename,
                    True,
                    csv_format,
                    file_format=file_format,
                    use_dictreader=use_dictreader,
                )
            )
    return result + result_part


def check_cache(src_list, cache_writers):
    src_ids = set(src.data_id for src in src_list if src.is_final)
    dest_ids = cache_writers.cache_id_set()

    ids_too_many = dest_ids - src_ids
    if ids_too_many:
        print(
            "ERROR: Inconsistent ID set in source filenames and {}".format(
                cache_writers.cache_filelist.filename
            )
        )
        print("Too many in cache list: {}".format(", ".join(ids_too_many)))
        input("Press ENTER")

    total_length = sum(
        len(src)
        for src in src_list
        if src.is_final and src.data_id in cache_writers.cache_filelist
    )
    for dest in cache_writers.dest_L:
        cache_length = (
            file_line_number(open(dest.cache_filename, **dest.file_format)) - 1
        )
        if cache_length > total_length:
            print(
                "ERROR: Inconsistent cache lengths (sources: {}, cachefile {}: {})".format(
                    total_length, dest.cache_filename, cache_length
                )
            )
            input("Press ENTER")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    filename_fact = lambda x: os.path.join(
        r"\\SW-FRATHC-FIL01\USERS-DE$\T1063588\Desktop\csvtest", x
    )

    cache_writers = CacheWriters(
        CsvCacheList(filename_fact("cache_filelist.txt")),
        CsvDestWriter(filename_fact("dest.txt"), ["A", "B"]),
    )

    srcs = csv_source_part_regex(
        glob.glob(filename_fact("d*.csv")), "part", dict(delimiter=";")
    )

    check_cache(srcs, cache_writers)
    import sys

    sys.exit(0)
    for src in srcs:
        writers = cache_writers.writers(src.data_id, src.is_final)
        if writers is not None:
            for row in src:
                for writer in writers:
                    print(row)
                    writer.writerow(row)

    cache_writers.close()

    import itertools as itoo

    assert set(open(filename_fact("dest.txt"))) - set(["\n"]) == set(
        itoo.chain.from_iterable(
            list(open(filename)) for filename in glob.glob(filename_fact("d*.csv"))
        )
    ) - set(["\n"])
