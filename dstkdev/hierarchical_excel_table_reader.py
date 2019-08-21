import xlwings as xw
import pandas as pd
import itertools
import re


def val_empty(val):
    return val in [None, ""]


def table_values_to_dataframe(table_values, num_header_rows, num_index_cols):
    num_cols = len(table_values[0])
    num_rows = len(table_values)

    for nrow in range(num_header_rows):
        last_val = None

        for ncol in range(num_index_cols, num_cols):
            val = table_values[nrow][ncol]
            if not val_empty(val):
                last_val = val
            elif last_val is not None:
                table_values[nrow][ncol] = last_val

    for ncol in range(num_index_cols):
        last_val = None
        for nrow in range(num_header_rows, num_rows):
            val = table_values[nrow][ncol]
            if not val_empty(val):
                last_val = val
            elif last_val is not None:
                table_values[nrow][ncol] = last_val

    col_index_values = [row[num_index_cols:] for row in table_values[:num_header_rows]]
    col_index_default_names = (f"col{i}" for i in itertools.count(1))
    col_index_names = [
        table_values[i][num_index_cols - 1] for i in range(num_header_rows - 1)
    ] + [
        None
    ]  # columns cannot have a name for the deepest
    index_names = [
        name if not val_empty(name) else default_name
        for name, default_name in zip(col_index_names, col_index_default_names)
    ]

    if index_names is None:
        col_index = None
    elif len(col_index_values) >= 2:
        col_index = pd.MultiIndex.from_arrays(col_index_values, names=index_names)
    elif len(col_index_values) == 1:
        col_index = pd.Index(col_index_values[0], names=index_names)
    else:
        raise ValueError(f"{col_index_values}")

    row_index_values = [
        [row[ncol] for row in table_values[num_header_rows:]]
        for ncol in range(num_index_cols)
    ]
    row_index_default_names = (f"idx{i}" for i in itertools.count(1))
    row_index_names = [
        table_values[num_header_rows - 1][i] for i in range(num_index_cols)
    ]
    index_names = [
        name if not val_empty(name) else default_name
        for name, default_name in zip(row_index_names, row_index_default_names)
    ]
    row_index = (
        pd.MultiIndex.from_arrays(row_index_values, names=index_names)
        if index_names
        else None
    )

    df = pd.DataFrame(
        [row[num_index_cols:] for row in table_values[num_header_rows:]],
        columns=col_index,
        index=row_index,
    )

    corner_cells = [
        row[: num_index_cols - 1] for row in table_values[: num_header_rows - 1]
    ]  # -1 since other entries are index names
    if set(cell for row in corner_cells for cell in row) <= {"", None}:
        corner_cells = None

    return df, corner_cells


def read_excel(
    excel_name, sheet_name, table_excel_range, num_header_rows=1, num_index_cols=0
):
    wb = xw.Book(excel_name)
    sht = wb.sheets[sheet_name]
    rng = sht.range(table_excel_range)
    table_values = rng.value
    wb.close()

    res = table_values_to_dataframe(
        table_values, num_header_rows=num_header_rows, num_index_cols=num_index_cols
    )
    return res


if __name__ == "__main__":
    from pathlib import Path

    # from sqlalchemy import create_engine    # Only for writing to a database

    data_dir = Path(r"E:\Usecases\uc150420\Daten VWN")
    # engine = create_engine(
    #    "postgresql://uc150420:T#Cross19@postgres9.vwdl.lan:5432/usecases"
    # )

    htg_dfs = []

    for file_basename, table_excel_range in [
        ("Kopie von Potenzialtool_HTGs_DE_20181022.xlsm", "G15:DZ1407"),
        ("Kopie von Potenzialtool_HTGs_EUR_20181022.xlsm", "G15:DZ10137"),
        ("Kopie von Potenzialtool_HTGs_INT_20181022.xlsm", "G15:DZ12396"),
    ]:
        file_name = data_dir.joinpath("HTG 2012-18", file_basename)

        # Read in Excel to a proper dataframe with (multi) indices
        carpark_df, corner_cells = read_excel(
            excel_name=str(file_name),
            sheet_name="Table",
            table_excel_range=table_excel_range,
            num_header_rows=3,
            num_index_cols=4,
        )

        # Reshape
        df2 = (
            carpark_df["Vor NR"]
            .stack("Kalenderjahr")
            .reset_index()
            .rename(columns={"idx2": "Land", "idx4": "HTG"})
        )

        htg_dfs.append(df2)

    htg_df = pd.concat(htg_dfs)

    htg_df.columns = htg_df.columns.map(lambda x: re.sub(r"\(|\)", "", x))
    htg_df.reset_index(drop=True, inplace=True)
    htg_df = htg_df.query("Kalenderjahr!='Gesamtergebnis'")
    htg_df["Kalenderjahr"] = htg_df["Kalenderjahr"].apply(pd.to_numeric)
    htg_df["BG / Umsatz VWN"] = htg_df["BG / Umsatz VWN"].apply(
        pd.to_numeric, errors="coerce"
    )
    for col in ["Marktgebiet", "Land", "Homogene Gruppe A", "HTG"]:
        htg_df[col] = htg_df[col].astype("category")

    htg_df.to_pickle(data_dir.joinpath("umsatz_anzahl.pkl"))

    file_basename = "Car Parc PR67.0.xlsm"
    table_excel_range = "A2:AB12627"
    file_name = data_dir.joinpath("Carpark 2011-18", file_basename)

    carpark_df, corner_cells = read_excel(
        excel_name=str(file_name),
        sheet_name="Roh-Daten",
        table_excel_range=table_excel_range,
        num_header_rows=1,
        num_index_cols=0,
    )

    carpark_df.columns = carpark_df.columns.map("_".join)  # to remove unneed multiindex
    carpark_df.reset_index(drop=True, inplace=True)

    for col in ["MARKT_NAME", "MARKE", "FZG_KLASSE"]:
        carpark_df[col] = carpark_df[col].astype("category")

    carpark_df["JAHR"] = carpark_df["JAHR"].apply(pd.to_numeric)

    carpark_df.to_pickle(data_dir.joinpath("carpark.pkl"))

    carpark_df.rename(
        columns={"MARKT_NAME": "land", "JAHR": "jahr", "Gesamt": "carpark"},
        inplace=True,
    )

    htg_df1 = htg_df.rename(
        columns={
            "Umsatz VWN": "umsatz",
            "Alternative Menge VWN": "menge",
            "Homogene Gruppe A": "htg_name",
            "HTG": "htg",
            "Land": "land",
            "Kalenderjahr": "jahr",
        }
    )

    htg_df1 = htg_df1.query(
        "htg_name!='Ergebnis' and htg!='#' and jahr!='Gesamtergebnis'"
    )

    htg_df1.to_pickle("umsatz_anzahl_clean.pkl")

    carpark_land = carpark_df.groupby(["land", "jahr"], as_index=False).sum()

    htg_df2 = htg_df1.merge(carpark_land[["land", "jahr", "carpark"]], how="left")

    htg_df2["land"] = htg_df2["land"].astype("category")

    htg_df2.to_pickle(data_dir.joinpath("umsatz_menge_carpark_clean.pkl"))
