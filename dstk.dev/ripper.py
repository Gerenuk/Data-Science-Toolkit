import subprocess
import operator
import re

weka_jar_path = "../weka.jar"


def ripper_rules(train_data_file):
    """
    train_data_file as arff
    """
    out = subprocess.check_output("java -cp {} weka.classifiers.rules.JRip -no-cv -i -t {}".format(weka_jar_path, train_data_file), shell=True)
    return out.decode("utf8")


def rules(jrip_output):
    rules_text = []
    lines = iter(jrip_output.split("\n"))

    for line in lines:
        if line.startswith("===="):
            break

    for line in lines:
        if line.strip() == "":
            continue

        if line.startswith("Number of Rules"):
            break

        rules_text.append(line)

    rule_struct = [[(var, {"<=": operator.le, ">=": operator.ge}[op], var)
                    for var, op, val in re.findall("\((\w+) ([<>=]+) ([-0-9.]+)\)", line)
                    ]
                   for line in rules_text[:-2]
                   ]

    return rule_struct


def num_rule(row, rule_struct):
    for i, rule in enumerate(rule_struct):
        if all(op(row[var], val) for var, op, val in rule):
            return i
    return -1
