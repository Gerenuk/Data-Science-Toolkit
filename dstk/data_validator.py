import argparse
import csv
import logging
import re
import sys

import pandas as pd
import yaml
from cytoolz import dissoc

"""
Calling script as
python data_validator.py -r filename=testdata.csv -w filename=testdata.pkl -c data.yml

Other ideas:
* add min/max rows check
* currently checking whether all rows have all needed columns isn't clean
* care when rows are non-string (check code!)
"""

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

type_caster_map = {"int": int,
                   "str": str,
                   "float": float,
                   }


def regex_checker(val, regex):   # Processors can modify values or throw exception
    match = re.fullmatch(regex, val)
    if not match:
        raise ParseError("Value {} failed to match regex {}".format(val, regex))
    return val

processor_map = {"regex": regex_checker,
                 }


class ParseError(Exception):
    pass


parser = argparse.ArgumentParser(description="CSV argument parser")
parser.add_argument("-r", help="Reader parameters in format var1=val1,var2=val2,...")
parser.add_argument("-w", help="Writer parameters in format var1=val1,var2=val2,...")
parser.add_argument("-c", help="YAML config file")

args = parser.parse_args()

logger.debug("Parameters {} passed to script".format(args))

config_filename = args.c

try:
    yaml_config = yaml.load(open(config_filename))

    reader_kwargs = yaml_config["reader"]
    writer_kwargs = yaml_config["writer"]  # Apart from filename not much used for pandas.to_pickle
    columns = yaml_config["columns"]

    logger.debug("Column specs are {}".format(columns))

    reader_kwargs._update({k: v for k, v in [k_v.split("=") for k_v in args.r.split(",")]})
    writer_kwargs._update({k: v for k, v in [k_v.split("=") for k_v in args.w.split(",")]})

    logger.debug("Reader arguments are {}".format(reader_kwargs))
    logger.debug("Writer arguments are {}".format(writer_kwargs))

    input_filename = reader_kwargs["filename"]
    input_file = open(input_filename, encoding=reader_kwargs.get("encoding"))
    logger.info("Reading file {}".format(input_filename))
    reader = csv.DictReader(input_file, **dissoc(reader_kwargs, "filename", "encoding"))

    column_names = [c["name"] for c in columns]
    data_column_names = reader.fieldnames
    if data_column_names != column_names:   # column order is strict
        raise ParseError("Column names stated do not match column names in data.\n" +
                         "Stated: {}\n".format(", ".join(map(str, column_names))) +
                         "In data: {}".format(", ".join(map(str, data_column_names)))
                         )

    column_processors = {column["name"]: column["processing"] for column in columns}
    column_types = {column["name"]: column["type"] for column in columns}

    column_names_set = set(column_names)

    processed_data = []
    for row_idx, row in enumerate(reader):
        processed_row = {}
        if row.keys() != column_names_set:
            raise ParseError("Row name do not match" +
                             "Missing rows: {}".format(", ".join(sorted(column_names_set - row.keys()))) +
                             "Excess rows: {}".format(",".join(row.keys() - column_names_set))
                             )

        for col_name, val in row.items():
            try:
                for processor in column_processors[col_name]:
                    val = processor_map[processor["name"]](val, **dissoc(processor, "name"))

                val = type_caster_map[column_types[col_name]](val)
            except Exception as exc:
                raise ParseError("Failed at row {} and column {}\n{}".format(row_idx, col_name, str(exc)))
            processed_row[col_name] = val
        processed_data.append(processed_row)

    processed_data = pd.DataFrame(processed_data)  # Would need to confirm types here. But final output probably parquet anyway

    output_filename = writer_kwargs["filename"]
    logger.info("Writing file {}".format(output_filename))
    processed_data.to_pickle(output_filename)  # Writer arguments actually not used

    logger.info("DONE")

except Exception as exc:
    logger.error("Data validation failed with:\n{}".format(exc))
    sys.exit(1)

