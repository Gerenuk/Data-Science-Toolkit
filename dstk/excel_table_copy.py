from xlwings import *
import pandas as pd
import os
import string
import shutil


class ExcelTTInput:
    def __init__(self, filename):
        """
        filename: Datei aus der die Tabelle eingelesen werden sollten
        """
        self.workbook = Workbook(filename)

    def horizontal_cellnum(self, sheet, col, row):
        return Range(sheet, (row, col)).horizontal.shape[1]

    def vertical_cellnum(self, sheet, col, row):
        return Range(sheet, (row, col)).vertical.shape[0]

    def read_table(self, sheet, col, row, header_levels=1, col_title_shift=1, col_num=None, row_num=None,
                   col_num_diff=0, row_num_diff=0):
        """
        Einlesen einer "hierarchisch" aufgebauten Tabelle in eine pandas.Series Datenstruktur
           | �21       | �22 ...
           | �11 | �12 | �13 | �14 ...
        ---------------------------
        Z1 | W1    W2    W3   W4
        Z2 | W5    W6    W7   W8
        
        Die �21, �22, usw Spalten sollten nicht verbunden sein, da sonst die Z�hlung durcheinander ger�t.
        Das Skript erg�nzt die �berschriften automatisch zu | �21  �21 | �22 �22 | .., aber wenn es auch
        schon genauso ausgef�llt ist, dann geht es auch.
        
        Normalerweise sollten die �berschriftenebenen sich wiederholen (�11=�13, �12=�14, ...), aber das Skript
        braucht das nicht.
        
        col/row gibt die Position (als Zahl) der ersten Zelle der tiefsten �berschriftenebene an (hier �11)
        header_levels gibt die Tiefe der �berschriftenebenen an (hier header_levels=2, da �1* und �2* da sind).
            Einfache Tabellen haben das voreingestellte header_levels=1
        col_title_shift gibt die Spaltendifferenz zwischen W1 und Z1 an. Man kann es verwenden wenn zwischen W1 und Z1
            Extraspalten sind, die nicht mit eingelesen werden sollen
        col_num/ row_num gibt die Anzahl der Spalten und Datenzeilen der Tabellen an. Sind diese Werte nicht gesetzt,
            werden sie automatisch bestimmt indem geschaut wird wie weit �11-�12-�13-.. und Z1-Z2-.. l�ckenlos gehen (�hnlich
            wie in Excel)
        col_num_diff, row_num_diff wird auf die Gesamtspalten/Datenzeilenanzahl addiert. Damit kann man z.B. die Anzahl der Spalten
            um -1 reduzieren (nachdem die Gesamtspaltenanzahl automatisch bestimmt wurde). So w�rde  man Extraspalten am Ende der Daten
            (z.B. Gesamtsummen die nicht zu den Daten geh�ren) ausschlie�en
            
        Das Ergebnis dieser Funktion ist eine pandas.Series wo W* die Werte sind und die Zeilen und �berschriften im Index mit dem
        Namen "rowname", "header1", "header2", ...
        Mehrere Ebenen bei den Zeilen werden momentan nicht unterst�tzt
        
        Im Beispiel w�re das Ergebnis:
        rowname header1 header2
        Z1      �11     �21      W1
        Z1      �12     �21      W2
        Z1      �13     �22      W3
        Z1      �14     �22      W4
        Z2      �11     �21      W5
        Z2      �12     �21      W6
        Z2      �13     �22      W7
        Z2      �14     �22      W8
   
        Diese Daten k�nnen nun mit Pandas Funktionen transformiert werden und in andere Excels rausgeschrieben.
        """
        self.workbook.set_current()
        # print("Reading {} sheet {}".format(self.workbook, sheet))

        if col_num is None:
            col_num = self.horizontal_cellnum(sheet, col, row) + col_num_diff

        if row_num is None:
            row_num = self.vertical_cellnum(sheet, col - col_title_shift, row + 1) + row_num_diff

        headers = [Range(sheet, (row - shift, col), (row - shift, col + col_num - 1)).value for shift in
                   reversed(range(header_levels))]
        row_titles = Range(sheet, (row + 1, col - col_title_shift), (row + row_num, col - col_title_shift)).value
        row_datas = Range(sheet, (row + 1, col), (row + row_num, col + col_num - 1)).value

        for h in headers:
            for i in range(1, len(h)):
                if h[i] is None:
                    h[i] = h[i - 1]

        index = []
        data = []

        for row_title, row_data in zip(row_titles, row_datas):
            for point, *col_titles in zip(row_data, *headers):
                index.append(tuple([row_title] + col_titles))
                data.append(point)

        series = pd.Series(data, index=pd.MultiIndex.from_tuples(index,
                                                                 names=["rowname"] + ["header{}".format(i) for i in
                                                                                      range(1, header_levels + 1)]))
        return series

    def read_value(self, sheet, col, row, col_title_shift=1, row_num=None, row_num_diff=0):
        """
        Diese Methode arbeitet �hnlich wie read_table, nur liest sie nur eine Spalte ein und erzeugt deshalb keine 
        �berschriften index levels im Ergebnis. Das kann n�tzlich sein, wenn es eine einzelne Spalte mit einem Wert
        pro Zeile gibt. z.B.
           | .. | Gesamt
        -----------------
        Z1 | .. | W1
        Z2 | .. | W2
        
        Das Ergebnis ist die Pandas.Series
        rowname  
        Z1       W1
        Z2       W2
        
        Mit index broadcasting in Pandas k�nnen solche Daten an die read_table Tabellen (welche mehr Ebenen haben) drangespielt
        werden.
        """
        if row_num is None:
            row_num = Range(sheet, (row + 1, col - col_title_shift)).vertical.shape[0] + row_num_diff

        row_titles = Range(sheet, (row + 1, col - col_title_shift), (row + row_num, col - col_title_shift)).value
        row_datas = Range(sheet, (row + 1, col), (row + row_num, col)).value

        index = []
        data = []

        for row_title, row_data in zip(row_titles, row_datas):
            index.append(tuple([row_title]))
            data.append(row_data)

        series = pd.Series(data, index=pd.MultiIndex.from_tuples(index, names=["rowname"]))
        return series


class ExcelTTOutput:
    def __init__(self, filename):
        self.workbook = Workbook(filename)

    def write_rows(self, sheet, col, row, data):
        """
        Schreibt 2D Daten (z.B. verschachtelte Liste) an eine bestimmt Position.
        """
        self.workbook.set_current()
        Range(sheet, (row, col)).value = data

    def save(self, filename=None):
        # self.workbook.xl_app.ActiveWorkbook.SaveAs(filename)
        self.workbook.save(filename)

    def close(self):
        self.workbook.close()


class ExcelTemplate:
    """
    Implementiert einen Context, der ein Template-Excel kopiert, dabei Platzhalter im Namen ersetzt und
    dann eine ExcelTTOutput Klasse zur�ckgibt.
    """
    def __init__(self, template_filename, name_map, dest_dir=None):
        self.template_filename = template_filename
        self.dest_dir = dest_dir if dest_dir is not None else os.path.dirname(template_filename)
        self.name_map = name_map
        self.excel_workbook = None

    def __enter__(self):
        basename = os.path.basename(self.template_filename)
        output_filename = os.path.join(self.dest_dir, string.Template(basename).substitute(self.name_map))
        shutil.copy(self.template_filename, output_filename)
        self.excel_workbook = ExcelTTOutput(output_filename)
        return self.excel_workbook

    def __exit__(self, exc_type, exc_value, traceback):
        self.excel_workbook.save()
        self.excel_workbook.close()