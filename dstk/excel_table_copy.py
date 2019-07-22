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
           | Ü21       | Ü22 ...
           | Ü11 | Ü12 | Ü13 | Ü14 ...
        ---------------------------
        Z1 | W1    W2    W3   W4
        Z2 | W5    W6    W7   W8
        
        Die Ü21, Ü22, usw Spalten sollten nicht verbunden sein, da sonst die Zählung durcheinander gerät.
        Das Skript ergänzt die Überschriften automatisch zu | Ü21  Ü21 | Ü22 Ü22 | .., aber wenn es auch
        schon genauso ausgefüllt ist, dann geht es auch.
        
        Normalerweise sollten die Überschriftenebenen sich wiederholen (Ü11=Ü13, Ü12=Ü14, ...), aber das Skript
        braucht das nicht.
        
        col/row gibt die Position (als Zahl) der ersten Zelle der tiefsten Überschriftenebene an (hier Ü11)
        header_levels gibt die Tiefe der Überschriftenebenen an (hier header_levels=2, da Ü1* und Ü2* da sind).
            Einfache Tabellen haben das voreingestellte header_levels=1
        col_title_shift gibt die Spaltendifferenz zwischen W1 und Z1 an. Man kann es verwenden wenn zwischen W1 und Z1
            Extraspalten sind, die nicht mit eingelesen werden sollen
        col_num/ row_num gibt die Anzahl der Spalten und Datenzeilen der Tabellen an. Sind diese Werte nicht gesetzt,
            werden sie automatisch bestimmt indem geschaut wird wie weit Ü11-Ü12-Ü13-.. und Z1-Z2-.. lückenlos gehen (ähnlich
            wie in Excel)
        col_num_diff, row_num_diff wird auf die Gesamtspalten/Datenzeilenanzahl addiert. Damit kann man z.B. die Anzahl der Spalten
            um -1 reduzieren (nachdem die Gesamtspaltenanzahl automatisch bestimmt wurde). So würde  man Extraspalten am Ende der Daten
            (z.B. Gesamtsummen die nicht zu den Daten gehören) ausschließen
            
        Das Ergebnis dieser Funktion ist eine pandas.Series wo W* die Werte sind und die Zeilen und Überschriften im Index mit dem
        Namen "rowname", "header1", "header2", ...
        Mehrere Ebenen bei den Zeilen werden momentan nicht unterstützt
        
        Im Beispiel wäre das Ergebnis:
        rowname header1 header2
        Z1      Ü11     Ü21      W1
        Z1      Ü12     Ü21      W2
        Z1      Ü13     Ü22      W3
        Z1      Ü14     Ü22      W4
        Z2      Ü11     Ü21      W5
        Z2      Ü12     Ü21      W6
        Z2      Ü13     Ü22      W7
        Z2      Ü14     Ü22      W8
   
        Diese Daten können nun mit Pandas Funktionen transformiert werden und in andere Excels rausgeschrieben.
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
        Diese Methode arbeitet ähnlich wie read_table, nur liest sie nur eine Spalte ein und erzeugt deshalb keine 
        Überschriften index levels im Ergebnis. Das kann nützlich sein, wenn es eine einzelne Spalte mit einem Wert
        pro Zeile gibt. z.B.
           | .. | Gesamt
        -----------------
        Z1 | .. | W1
        Z2 | .. | W2
        
        Das Ergebnis ist die Pandas.Series
        rowname  
        Z1       W1
        Z2       W2
        
        Mit index broadcasting in Pandas können solche Daten an die read_table Tabellen (welche mehr Ebenen haben) drangespielt
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
    dann eine ExcelTTOutput Klasse zurückgibt.
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