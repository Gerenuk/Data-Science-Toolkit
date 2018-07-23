import pandas as pd
from operator import itemgetter
from itertools import groupby
from cytoolz import pluck
from collections import Counter, defaultdict
import csv
import os


class FrequencyTracker:
    def __init__(self, varname, varfunc, extra_varnames=[], none_value=None):
        self.varname = varname
        self.varfunc = varfunc
        self.extra_varnames = extra_varnames
        self.counter = Counter()

        self.last_rarest_row = None
        self.last_row = defaultdict(lambda: none_value)

        self.value = None

    def add(self, row):
        val = self.varfunc(row)
        self.counter[val] += 1
        total_num = sum(self.counter.values())

        rarest_val, rarest_count = self.counter.most_common()[-1]
        rarest_freq = rarest_count / total_num

        if self.last_rarest_row is None or val == rarest_val:
            self.last_rarest_row = row

        val_last = self.varfunc(self.last_row)
        result = {self.varname + "_freq": self.counter[val] / total_num,
                  self.varname + "_israrest": int(rarest_val == val),
                  self.varname + "_freqrarest": rarest_count / total_num,
                  self.varname + "_islast": int(val == val_last),
                  self.varname + "_freqlast": self.counter[val_last] / total_num,
                  }

        for extra_var in self.extra_varnames:
            result.update({self.varname + "_rarest_" + extra_var: self.last_rarest_row[extra_var],
                           self.varname + "_last_" + extra_var: self.last_row[extra_var],
                           })

        self.value = result

        return result


class TimestampSameCustomer:
    def __init__(self):
        self.last_time = 0
        self.cur_time = 0

    def add(self, row):
        if row["same_customer"]:
            self.last_time = self.cur_time
            self.cur_time = row["timestamp"]

    @property
    def value(self):
        return {"timestamp_samecustomer": self.last_time}


def trans_iter():
    for kdnr, groups in groupby(map(dict, pluck(1, trans_dat.iterrows())), key=itemgetter("kdnr")):
        aggs = (
            [FrequencyTracker(varname, varfunc, ["timestamp"], 0) for varname, varfunc in
             [("land", itemgetter("ref_land")),
              ("kanal", lambda x: str(
                  x["UTU5_EINGABE_NAME"])[
                                  :5])]] +
            [TimestampSameCustomer()])
        for row in groups:
            for agg in aggs:
                agg.add(row)
                row.update(agg.value)
            yield row


def tuple_dict_combiner(rows, included):
    result_row = {}
    # conflict = set()
    for i, row in enumerate(rows):
        for k, v in row.items():
            if k not in result_row:
                result_row[k] = v
            else:
                pass
                # DO NOT recreate additional variables
                # new_name=k+str(i)
                # if k not in conflict:
                #    conflict.add(k)
                #    print("Conflicting variable {}. Renaming to {}".format(k, new_name))
                # result_row[new_name]=v
    return result_row, included


class MergeColumn:
    """
    Maintains self.key_row with self.cur_key>min_key
    """

    def __init__(self, data, key_func=None, missing_row=None, equal_key_stepwise=False):
        self.data = iter(data)
        self.key_func = key_func if key_func is not None else lambda x: x

        self.equal_key_stepwise = equal_key_stepwise

        self.last_row = missing_row
        self.last_key = None
        self.key_row = next(self.data)
        self.cur_key = self.key_func(self.key_row)

        self.missing_row = missing_row

    @property
    def min_key(self):
        """
        min_key to determine least of all current keys
        """
        return self.cur_key

    def row(self, min_key):
        # print("K", min_key, self.cur_key)

        try:
            while self.cur_key is not None and self.cur_key <= min_key:
                self.last_row = self.key_row
                self.key_row = next(self.data)
                self.last_key = self.cur_key
                self.cur_key = self.key_func(self.key_row)
                assert self.last_key <= self.cur_key, "Keys not sorted {} <= {}".format(self.last_key, self.cur_key)
                if self.equal_key_stepwise:
                    break
        except StopIteration:
            self.last_row = self.key_row
            self.last_key = self.cur_key
            self.key_row = None
            self.cur_key = None

        return self._cur_row(min_key)

    def _cur_row(self, min_key):
        if self.last_key == min_key:
            return (self.last_row, True)
        else:
            return (self.missing_row, False)


class MergeColumnPart(MergeColumn):
    """
    needs tuples as keys
    currently will not create output for unmatched
    """

    @property
    def min_key(self):
        return None

    def _cur_row(self, min_key):
        if self.last_key is None:
            return self.last_row

        key_len = min(len(self.last_key), len(min_key))

        if self.last_key[:key_len] == min_key[:key_len]:
            return (self.last_row, True)
        else:
            return (self.missing_row, False)


class MergeColumnLast(MergeColumn):
    """
    currently does not create entries for unmatched
    """

    @property
    def min_key(self):
        return None

    def _cur_row(self, min_key):
        if self.last_key is not None:
            return self.last_row, True
        else:
            return self.last_row, False


class MergeColumnLastPart(MergeColumn):
    def __init__(self, data, key_func=None, missing_row=None, equal_key_stepwise=False, len_key=1):
        super().__init__(data, key_func, missing_row, equal_key_stepwise)
        self.len_key = len_key

    @property
    def min_key(self):
        return None

    def _cur_row(self, min_key):
        if self.last_key is not None and self.last_key[:self.len_key] == min_key[:self.len_key]:
            return (self.last_row, True)
        else:
            return (self.missing_row, False)


def merge(*merge_cols, required=lambda x: True, combiner=lambda row, inc: tuple([row, inc])):
    keys = [mc.min_key for mc in merge_cols]
    while 1:
        if all(mk is None for mk in keys):
            break
        min_key = min(k for k in keys if k is not None)

        rows=[]
        includeds=[]

        for i, mc in enumerate(merge_cols):
            try:
                row, included=mc.row(min_key)
                rows.append(row)
                includeds.append(included)
            except Exception as e:
                print("Merge of row number {} failed with error {}".format(i, e))
                raise

        rows, included = zip(*[mc.row(min_key) for mc in merge_cols])
        if required(included):
            yield combiner(rows, included)

        new_keys = [mc.min_key for mc in merge_cols]
        # assert all(old_key <= new_key for old_key, new_key in zip(keys, new_keys) if
        #           old_key is not None and new_key is not None), "Keys not sorted: {} <= {}".format(keys, new_keys)
        keys = new_keys


def missing_row(colnames):
    row = {}
    for colname in colnames:
        if "_freq" in colname:
            val = 1
        elif colname.endswith("_islast"):
            val = 1
        elif colname.endswith("_israrest"):
            val = 0
        elif "_timesince_" in colname:
            val = 1e9
        elif colname == "EID_kdnr_counts":
            val = 1
        elif "_count" in colname:
            val = 0
        elif colname == "session_time":
            val = 0
        elif colname in ["clicktype", "content"]:
            val = 0
        elif colname.startswith("timestamp_"):
            val = 0
        elif colname == "KZVH_AENDERUNGSZEITPUNKT":
            val = 0
        elif colname.endswith("_timestamp"):
            val = 0
        elif colname == "email_provider":
            val = "nomail"
        elif colname in ("trans_f_time_since_EX_to_DI_m300", "trans_f_time_since_AK_to_DI_m300"):
            val = 1000000
        elif colname in ("trans_f_amount_EX_to_DI_m300", "trans_f_amount_AK_to_DI_m300"):
            val = 0
        elif colname == "timestamp":
            val = 0
        else:
            val = float("nan")
        row[colname] = val
    return row


def makeint(x):
    try:
        return int(x)
    except ValueError:
        print("Error with value {!r}".format(x))
        return 0


if __name__ == '__main__':
    import os

    #base_dir = "/BIGDATA/home/asuchane/Projects/FRA/data"

    trans_filename = os.environ["INPUT0"] # os.path.join(base_dir, "clean_tim_features.tab")
    web_filename = os.environ["INPUT1"] # os.path.join(base_dir, "Webtrekk_Sessions_merged_processed.tab")
    internet_filename = os.environ["INPUT2"] # os.path.join(base_dir, "Internet logins processed.tab")
    mobile_filename = os.environ["INPUT3"] # os.path.join(base_dir, "Mobile logins processed.tab")
    refbank_filename = os.environ["INPUT4"] # os.path.join(base_dir, "reference_bank_change.tab")
    cust_email_filename = os.environ["INPUT8"] # os.path.join(base_dir, "kunde_email_provider.tab")
    trans_internal_filename = os.environ["INPUT5"] # os.path.join(base_dir, "trans_internal_to_DI_features.tab")
    #eid_count_filename = os.environ["INPUT7"] # os.path.join(base_dir, "EID_counts.tab")
    grey_black_ref_filename = os.environ["INPUT6"] # os.path.join(base_dir, "grey_black_ref_fixed.tab")
    timesorted_filename = os.environ["INPUT7"] # os.path.join(base_dir, "timesorted_features.tab")
    #kunden_email_filename = os.environ["INPUT5"] # os.path.join(base_dir, "kunde_email_provider.tab")

    trans_file = os.environ["OUTPUT0"] # os.path.join(base_dir, "clean_final_features.tab")

    trans_dat = pd.read_csv(trans_filename, delimiter="\t", encoding="cp1252",
                            low_memory=False)  # , compression="gzip")
    web_dat = pd.read_csv(web_filename, delimiter="\t")
    internet_dat = pd.read_csv(internet_filename)
    mobile_dat = pd.read_csv(mobile_filename)
    refbank_dat = pd.read_csv(refbank_filename, delimiter="\t")
    # cust_email = pd.read_csv(cust_email_filename, delimiter="\t")
    # trans_internal = pd.read_csv(trans_internal_filename, delimiter="\t")
    #eid_counts = pd.read_csv(eid_count_filename, sep="\t")

    trans_dat.sort(["kdnr", "timestamp"], inplace=True)

    refbank_dat.rename(columns={"timestamp_1970": "timestamp_refbank"}, inplace=True)
    internet_dat.rename(columns={"timestamp": "timestamp_internet"}, inplace=True)
    mobile_dat.rename(columns={"timestamp": "timestamp_mobile"}, inplace=True)
    web_dat.rename(columns={"timestamp": "timestamp_web"}, inplace=True)
    #eid_counts.rename(columns={"EID_count": "EID_kdnr_counts"}, inplace=True)
    trans_dat.dropna(subset=["timestamp"], inplace=True)
    web_dat.dropna(subset=["timestamp_web"], inplace=True)

    trans_dat["empf_kdnr"] = trans_dat.UTU5_EMPF_ZPFL_KONTO.map(
        dict(trans_dat.ix[:, ["UTU5_AUFTR_ZEMPF_KONTO", "kdnr"]].itertuples(False)))
    trans_dat["same_customer"] = trans_dat.empf_kdnr == trans_dat.kdnr



    # grey_list = csv.DictReader(open(grey_black_ref_filename, newline="", encoding="utf8"), delimiter="\t")
    grey_list = pd.read_csv(grey_black_ref_filename, sep="\t")
    grey_list.sort(["kdnr", "EINGABE_DATETIME_1970"], inplace=True)

    timesorted_data = csv.DictReader(open(timesorted_filename, newline="", encoding="utf8"), delimiter="\t")

    data = merge(MergeColumn(trans_iter(),  # pluck(1,trans_dat.iterrows()),
                             itemgetter("kdnr", "timestamp"),
                             equal_key_stepwise=True,
                             missing_row=missing_row(trans_dat.columns)),
                 MergeColumnLastPart(pluck(1, web_dat.iterrows()),
                                     itemgetter("KDNR", "timestamp_web"),
                                     missing_row=missing_row(web_dat.columns),
                                     len_key=1),
                 MergeColumnLastPart(pluck(1, internet_dat.iterrows()),
                                     itemgetter("kdnr", "timestamp_internet"),
                                     missing_row=missing_row(internet_dat.columns),
                                     len_key=1),
                 MergeColumnLastPart(pluck(1, mobile_dat.iterrows()),
                                     itemgetter("kdnr", "timestamp_mobile"),
                                     missing_row=missing_row(mobile_dat.columns),
                                     len_key=1),
                 MergeColumnLastPart(pluck(1, refbank_dat[refbank_dat.KZVH_NUTZUNGSTYP == 12].ix[:,
                                              ["KZVH_KUNDEN_NR", "timestamp_refbank"]].iterrows()),
                                     itemgetter("KZVH_KUNDEN_NR", "timestamp_refbank"),
                                     missing_row=missing_row(["KZVH_KUNDEN_NR", "timestamp_refbank"]),
                                     len_key=1),
                 #MergeColumnLastPart(pluck(1, eid_counts.iterrows()),
                 #                    itemgetter("KDNR", "timestamp"),
                 #                    missing_row=missing_row(eid_counts.columns),
                 #                    len_key=1,
                 #                    ),
                 # MergeColumnPart(pluck(1, cust_email.iterrows()),
                 #                lambda x: (x["kdnr_0001"],),
                 #                missing_row=missing_row(cust_email.columns)
                 #                ),
                 # MergeColumn(pluck(1, trans_internal.iterrows()),
                 #            itemgetter("kdnr_0001", "timestamp"),
                 #            missing_row=missing_row(trans_internal.columns),
                 #            ),
                 MergeColumn(pluck(1, grey_list.iterrows()),
                             lambda x: (int(x["kdnr"]), makeint(x["EINGABE_DATETIME_1970"])),
                             missing_row=missing_row(grey_list.columns),
                             ),
                 MergeColumn(timesorted_data,
                             lambda row: (int(row["kdnr"]), int(row["timestamp"])),
                             missing_row=missing_row(timesorted_data.fieldnames)
                             ),
                 required=itemgetter(0),
                 combiner=tuple_dict_combiner,
                 )

    #kunde_email_df = pd.read_csv(kunden_email_filename, sep="\t")
    #kunde_email = defaultdict(lambda: "noemail", zip(kunde_email_df.kdnr_0001, kunde_email_df.email_provider))

    with open(trans_file, "w", newline="", encoding="utf8") as f:
        writer = None

        for row in pluck(0, data):
            for timevar in ["timestamp_web", "timestamp_internet", "timestamp_mobile", "timestamp_refbank",
                            "timestamp_samecustomer", "kanal_rarest_timestamp", "land_rarest_timestamp"]:
                row[timevar + "_timesince"] = row["timestamp"] - row[timevar]

            if row["trans_c_day_of_week"] == 7: # remap so that week order is Monday -> Sunday and rules can split weekend
                row["trans_c_day_of_week"] = 0

            for c in list(row.keys()):
                if "m56" in c:
                    row["whitelist_" + c] = row[c] if row["same_customer"] == 0 else 0

                if "l35d" in c and "value" in c:
                    row["spike_1_" + c] = row["trans_f_transaction_amount"] / row[c] if row[c] > 100 else 0

            #row["email_provider"] = kunde_email[row["kdnr"]]
            row["channel5"] = row["UTU5_EINGABE_NAME"][:5]

            if writer is None:
                writer = csv.DictWriter(f, sorted(row.keys()), delimiter="\t")
                writer.writeheader()

            writer.writerow(row)
