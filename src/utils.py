from typing import List

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
