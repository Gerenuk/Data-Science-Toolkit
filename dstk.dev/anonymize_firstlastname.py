import re

default_nonanon_regex = r"\bgmbh\b|\bklinik| co kg\b| co\.|[a-z]+ [a-z]+ ag| cokg\b|krankenhaus|sagt danke|\bstadt [a-z]+"


class AnonFirstLastName:
    def __init__(
        self,
        blacklist,
        firstnames,
        lastnames,
        firstlastnames,
        nonanon_regex=default_nonanon_regex,
    ):
        self.blacklist = blacklist
        self.firstnames = firstnames
        self.lastnames = lastnames
        self.firstlastnames = firstlastnames
        self.nonanon_regex = re.compile(nonanon_regex, flags=re.I)
        self.word_regex = re.compile(
            r"\b[^\W0-9_]{2,}\b"
        )  # [^\W0-9_] effectively means "any letter", also Umlaut etc.
        self.replace_dict = {}  # maybe speedup with lookup

    def __call__(self, text):
        if self.nonanon_regex.search(text):
            return text

        names = set(self.word_regex.findall(text.lower()))

        if not (
            names & self.firstnames.value
            and names & self.lastnames.value
            and len(names & self.firstlastnames.value) > 1
        ):
            # latter condition is to ensure that not a single word found in both firstnames and lastnames triggers
            # anonymization (e.g. a word that can be both first and last name)
            return text

        to_replace = names & self.blacklist.value
        for replace_x in to_replace:
            replace_len = len(replace_x)
            if replace_len not in self.replace_dict:
                self.replace_dict[replace_len] = "X" * replace_len
            replace_text = self.replace_dict[replace_len]

            text = re.sub(r"\b" + replace_x + r"\b", replace_text, text, flags=re.I)
        return text


def anon_firstlastname_spark(df, blacklist, firstnames, lastnames, *columns_to_anon):
    from pyspark.sql.functions import udf, StringType, col

    sc = df._sc

    blacklist = set(map(str.lower, blacklist))
    firstnames = set(map(str.lower, firstnames))
    lastnames = set(map(str.lower, lastnames))

    blacklist_bc = sc.broadcast(blacklist)
    firstnames_bc = sc.broadcast(firstnames)
    lastnames_bc = sc.broadcast(lastnames)
    firstlastnames_bc = sc.broadcast(firstnames | lastnames)

    anonymizer = AnonFirstLastName(
        blacklist_bc, firstnames_bc, lastnames_bc, firstlastnames_bc
    )
    anonymizer_udf = udf(anonymizer.__call__, StringType())

    result = df
    for colname in columns_to_anon:
        print("Anonymizing column {}".format(colname))
        result = result.withColumn(
            colname + "_anonymized", anonymizer_udf(col(colname))
        ).drop(colname)

    return result
