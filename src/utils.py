import os
import pickle
from typing import Any, List
from numpy.lib.arraysetops import isin

import pandas as pd
from pandas import DataFrame

DATA_FOLDER = '../data/'
OUT_FOLDER = '../out/'


def verify_data(df: DataFrame) -> None:
    """
    Verifies the given dataframe.
    :param df: DataFrame
    :return: None
    """
    print(df.shape)
    print(df.head())


def drop_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    """
    Returns a new dataframe by dropping the given columns from the given
    dataframe.
    :param df: DataFrame
    :param columns: list[str]
    :return: DataFrame
    """
    return df.drop(columns, axis='columns')


def drop_indices(df: DataFrame, indices: List[int]) -> DataFrame:
    """
    Returns a new dataframe by dropping the given indices from the given
    dataframe.
    :param df: DataFrame
    :param indices: list[int]
    :return: DataFrame
    """
    return df.drop(indices, axis='index')


def save_to_pickle(obj: Any, name: str) -> None:
    """
    Pickles the given object.
    :param obj: object
    :param name: str
    :return: None
    """
    path = os.path.join(OUT_FOLDER, name)
    if isinstance(obj, DataFrame):  # DataFrame
        obj.to_pickle(path)
    else:  # Machine learning model
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    print(f'Object is written to {path}')


def read_from_pickle(name: str, is_dataframe: bool=True) -> DataFrame:
    """
    Reads an object from pickle.
    :param name: str
    :param is_dataframe: bool
    :return: DataFrame
    """
    path = os.path.join(OUT_FOLDER, name)
    if is_dataframe:  # DataFrame
        return pd.read_pickle(path)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)
