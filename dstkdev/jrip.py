__author__ = "asuchane"

import subprocess
import arff


def dump_arff(X, y=None, filename="data4jrip.arff"):
    if y is not None:
        data = {
            "attributes": [(c, "REAL") for c in X.columns]
            + [(y.name, list(map(str, y.unique())))],
            "data": [row + (str(y),) for row, y in zip(X.itertuples(False), y)],
            "description": "",
            "relation": "data4jrip",
        }
    else:
        data = {
            "attributes": [(c, "REAL") for c in X.columns],
            "data": [row for row in X.itertuples(False)],
            "description": "",
            "relation": "data4jrip",
        }

    with open(filename, "w") as f:
        f.write(arff.dumps(data))

    print("File {} written".format(filename))
    return filename


def jrip(*args, **kwargs):
    cl_params = " ".join(
        ["-" + a for a in args] + ["-{} {}".format(k, v) for k, v in kwargs.items()]
    )
    print(
        "Calling:",
        "java -cp /export/home/asuchane/Programs/weka-3-6-13/weka.jar weka.classifiers.rules.JRip "
        + cl_params,
    )
    output = subprocess.check_output(
        "java -cp /export/home/asuchane/Programs/weka-3-6-13/weka.jar weka.classifiers.rules.JRip "
        + cl_params,
        shell=True,
    )

    return output.decode()


if __name__ == "__main__":
    import pandas as pd
    from random import randint
    import random

    import os

    os.environ["CLASSPATH"] = "/export/home/asuchane/Programs/weka-3-6-13/weka.jar"

    random.seed(1)
    N = 1000
    X = pd.DataFrame(
        [(randint(0, 9), randint(0, 9)) for _ in range(N)], columns=["a", "b"]
    )
    y = pd.Series([randint(0, 1) for _ in range(N)], name="fraud")
    r = jrip(X, y)

    print(r)
