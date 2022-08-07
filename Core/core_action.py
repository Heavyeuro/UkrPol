import numpy as np
import pandas as pd

# reading file in CSV format
from pandas import DataFrame


def read_csv(file_name):
    return pd.read_csv(r'..\CsvStorage' + file_name)


# Replace each val in appropriate cols
def replace_val_in_cols(data_csv: DataFrame, column_list, old_value: str, new_value: str):
    for col in column_list:
        data_csv[col] = data_csv[col].replace(old_value, new_value)
    return data_csv


def replace_val_in_cols_except(data_csv: DataFrame, column: str, except_value: str, new_value: str):
    for val in data_csv[column]:
        if val != except_value:
            data_csv[column] = data_csv[column].replace(val, new_value)
    return data_csv


# replacing 0 to NaN in dataset
def null_to_NaN(X, except_list):
    cols_with_missing_val = detect_vals(X, 0, except_list).index[0]
    X = replace_val_in_cols(X, [cols_with_missing_val], 0, np.nan)
    return X


# Counting an amount of null values in dataset by column
def detect_vals(obj_to_describe, val, exclude_col=[]):
    # missing values by columns
    obj_to_describe = obj_to_describe.drop(exclude_col, axis=1)
    missing_val_count_by_column = (obj_to_describe.isin([val]).sum())
    return (missing_val_count_by_column)
