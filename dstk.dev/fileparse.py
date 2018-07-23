from collections import defaultdict, Counter
import csv

def fileparse(filename):
    fp=FileParse(filename)
    fp.parse_sep()
    fp.parse_rows()
    return fp


class FileParse:
    def __init__(self, filename):
        self.filename=filename
        self.sep_counts=None
        self.best_sep=None
        self.sep=None
        self.linecount=0

    def _open(self, *args, **kwargs):
        return open(self.filename, *args, **kwargs)

    def parse_sep(self):
        all_charcounts=defaultdict(list)
        with self._open() as f:
            for line in f:
                line=line[:-1]
                if line=="":
                    continue
                self.linecount+=1
                charcount=Counter(line)
                for key, count in charcount.items():
                #    if key in all_charcounts:
                        all_charcounts[key].append(count)
                #for key in list(all_charcounts.keys()):
                #    if key not in all_charcounts:
                #        del all_charcounts[key]
        print("Separator char found:")

        self.sep_counts={sep:Counter(counts) for sep, counts in all_charcounts.items()}
        self.best_sep="".join(sep for sep, _count in sorted(self.sep_counts.items(), key=lambda t:max(q*r for q,r in t[1].items()), reverse=True))

        print("Linecount {}".format(self.linecount))
        for sep in self.best_sep[:5]:
            print("{} : {}".format(sep, ", ".join("{}sep ({})".format(num, count if count!=self.linecount else "ALL") for num, count in self.sep_counts[sep].most_common(5))))

        print("Best separator {}".format(self.best_sep[:10]))
        self.sep=self.best_sep[0]

    def parse_rows(self, sep=None):
        if sep is None:
            sep=self.sep

        print("TEST COLUMN NUMBER")
        print("Using separator '{}'".format(sep))
        with self._open(newline="") as f:
            reader=csv.reader(f, delimiter=sep)
            row_lengths=Counter()
            col_lengths=[]
            for i, row in enumerate(reader):
                if i==0:
                    header=row
                row_lengths[len(row)]+=1
                col_lengths.append([len(el) for el in row])


        print("Row lengths ", ", ".join("{}length({})".format(length, count if count!=self.linecount else "ALL") for length, count in row_lengths.most_common()))
        for head, cols in zip(header, zip(*col_lengths)):
            print("{} length {}..{}".format(head, min(cols), max(cols)))


if __name__ == '__main__':
    fp=FileParse(r"F:\V\VT_Oberursel\RBV\KA\CRM_DM_Mafo\04_CMI\02_Aufgaben\04_Datenquellen\12_SAP_Reiseaufträge, Kundendaten\RA_Abzug_ab20110101_ALL_lines10000.csv")
    fp.parse_sep()
    fp.parse_rows()