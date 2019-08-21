import re

from functools import partial

try:
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
except ImportError:
    print("Spark not loaded")

from itertools import groupby
from cytoolz import pluck, partition
import re

compname_default_col_name = "gegenkonto_name"
match_stats_default_col_name = "matchstats"
match_default_col_name = "match"
cat_default_col_name = "categories"
PRIVATE_TAG = "PRIVATE"


def select_match_words(text, num_words):
    text = text.lower()

    for replace_text, new_text in [(".", ""),  # to compress "E.ON"
                                   ("ü", "ue"),
                                   ("ä", "ae"),
                                   ("ö", "oe"),
                                   ("ß", "ss"),
                                   ]:
        text = text.replace(replace_text, new_text)

    parts = re.split(r"(\w+)", text)
    words = [(word, preword) for preword, word in partition(2, parts)]

    # remove digits, but keep combination of letters and digit
    words = list(filter(lambda x: not x[0].isdigit(), words))

    if not words:
        return tuple()

    # these words are removed; Roman numerals not considered even though they appear
    if words[0][0] in {"erste", "zweite", "dritte", "vierte", "fuenfte",
                       "sechste", "visa"}:
        words = words[1:]

    words = _compress_letter_and_initabbr(words)

    result = []
    word_count = 0
    for word, preword in words:
        result.append(word)
        if word not in {"dr", "med", "der", "stadt", "und", "fuer", "of",
                        "the", "die", "das", "am", "deutsches", "deutsche", "deutscher",
                        "verein", "klink", "institut", "st",
                        } and len(word) >= 2 and preword != "-":
            # these words are not counted towards the min. num_words word_rule
            # words connected by dashes are taken as single words, but count only 1 towards the word count
            # -> "Max-Planck-Institut ABC" -> "max planck institut abc" instead of "max planck"
            word_count += 1
        if word_count == num_words:
            break

    if sum(map(len, result)) < 5:  # names too short don't match
        return tuple()

    return tuple(result)


def _compress_letter_and_initabbr(words):
    if not words:
        return words
    result = []
    for word_len, words_grp in groupby(words, lambda x: len(x[0])):  # compress single letter words
        if word_len == 1:
            result.append(("".join(pluck(0, words_grp)), None))
        else:
            result.extend(words_grp)

    if (result and 2 <= len(result[0]) <= len(result) - 1 and  # remove if first word is an abbrevation of following words, e.g. "ABC Aa Bb Cc Company"
            all(char == word[0] for char, word in zip(result[0][0], pluck(0, result[1:])))):
        result = result[1:]

    return result


def findmatch(text, word_match_tree, num_words, max_tx_skip):
    matches = []
    match_nextwordcount = []
    matchdict_idxs = []

    if text.count("XXX") >= 2:
        matches.append(PRIVATE_TAG)
        match_nextwordcount.append(0)
        matchdict_idxs.append(0)

    words = select_match_words(text, num_words=num_words)

    for skip in range(max_tx_skip + 1):
        cur_match = []
        cur_dict = word_match_tree

        for word in words[skip:]:
            if word not in cur_dict:
                break
            cur_match.append(word)
            cur_dict = cur_dict[word]

        if cur_match:
            matches.append(" ".join(cur_match))
            match_nextwordcount.append(len(cur_dict))
            matchdict_idxs.append(skip)

    num_tx_words = len(re.findall(r"\w+", text))
    tx_words = " ".join(words)
    return matches, match_nextwordcount, matchdict_idxs, num_tx_words, tx_words


def make_word_match_tree(matchlist):
    word_match_tree = {}
    for words in matchlist:
        cur_dict = word_match_tree
        for word in words:
            cur_dict = cur_dict.setdefault(word, {})
    return word_match_tree


def add_match_stats_col(df,
                        matchlist,
                        compname_col_name=compname_default_col_name,
                        match_stats_col_name=match_stats_default_col_name,
                        max_tx_skip=1,
                        num_words=2):
    word_match_tree = make_word_match_tree(matchlist)

    findmatch_partial = partial(findmatch,
                                word_match_tree=df._sc.broadcast(word_match_tree).value,
                                num_words=num_words,
                                max_tx_skip=max_tx_skip,
                                )

    findmatch_udf = udf(findmatch_partial, StructType([StructField("words", ArrayType(StringType())),
                                                       StructField("nextwordcount", ArrayType(IntegerType())),
                                                       StructField("idxs", ArrayType(IntegerType())),
                                                       StructField("num_words", IntegerType()),
                                                       StructField("tx_words", StringType()),
                                                       ]))
    return df.withColumn(match_stats_col_name, findmatch_udf(compname_col_name))


def filter_match(match_stats, exclude):
    words, numfollowwords, skips, totwords, tx_words = match_stats
    result = []
    for word, numfollowword, skip in zip(words, numfollowwords, skips):
        if (len(word) <= 2 or
                    #numfollowword > 15 or
                    word in exclude.value):
            continue
        if word == PRIVATE_TAG:
            return PRIVATE_TAG  # , ""
        if skip == 0:
            result.append(word)
        elif skip == 1 and len(word.split()) >= 2:
            result.append(word)
    # if not result:
    #    return "", ""
    return ";".join(sorted(result))  # , tx_words


def add_filtered_match_col(df, exclude=frozenset(), match_stats_col_name=match_stats_default_col_name,
                           match_col_name=match_default_col_name):
    filter_match_udf = udf(partial(filter_match, exclude=exclude), StringType(),
                           # StructType([StructField("match_suggestion", StringType()), StructField("tx_words", StringType())])
                           )
    return df.withColumn(match_col_name, filter_match_udf(match_stats_col_name))


def find_categories(text, cat_match):
    result = set()

    matches = text.split(";")
    for match in matches:
        if match == PRIVATE_TAG:
            result.add("PRIVATE")
            continue
        words = match.split()
        result.update(cat_match[tuple(words)])
    return list(result)


def add_category_col(df, cat_match, match_col_name=match_default_col_name, cat_col_name=cat_default_col_name):
    find_categories_partial = partial(find_categories, cat_match=df._sc.broadcast(cat_match).value)
    find_categories_udf = udf(find_categories_partial, ArrayType(StringType()))

    return df.withColumn(cat_col_name, find_categories_udf(match_col_name))
