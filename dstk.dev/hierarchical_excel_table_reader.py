import pandas as pd
import itertools
import xlwings as xw


def val_empty(val):
    return val in [None, ""]


def table_values_to_dataframe(table_values, num_header_rows, num_index_cols):
    """
    TODO:
    * Custom index names (mapping)
    * Warning on empty row/cols
    * Header/Index may be detached from data block
    """
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

    col_index = pd.MultiIndex.from_arrays(col_index_values, names=index_names) if index_names else None

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
    row_index = pd.MultiIndex.from_arrays(row_index_values, names=index_names) if index_names else None

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
