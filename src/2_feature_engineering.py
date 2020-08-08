import os

import pandas as pd
from pandas import DataFrame

from utils import OUT_FOLDER, verify_data


def _freq_of_locations(df: DataFrame):
    """
    Returns the frequency of locations in the given dataframe.
    :param df: DataFrame
    """
    freq_of_locs = df.groupby('location')['location'].agg(
        'count').sort_values(ascending=False)
    print(freq_of_locs)
    return freq_of_locs


def main():
    df = pd.read_pickle(os.path.join(OUT_FOLDER, 'data_cleaned.pkl'))

    # Consider the "location" column, which is a non-quantitative column

    print(len(df['location'].unique()))  # 1304

    # * ONE-HOT ENCODING *

    # To convert the 1304 distinct locations into quantitative features, we can
    # use ONE-HOT ENCODING, which means creating 1304 more location features.
    # => In this way, we would have too many dimensions, leading to "curse of
    #    dimensionality".

    # * DIMENSIONALITY REDUCTION *

    # Check the number of properties for each location
    freq_of_locs = _freq_of_locations(df)
    # We can see that most of the locations only have a small number of
    # properties.

    # => We can set up some threshold T, to aggregate all the locations with
    #    <= T properties to a location category "other".
    # => In this way, we can significantly reduce the number of dimensions.
    freq_of_locs_less_than_10 = freq_of_locs[freq_of_locs <= 10]
    print(len(freq_of_locs_less_than_10))  # 1063
    df['location'] = df['location'].apply(
        lambda x: 'others' if x in freq_of_locs_less_than_10 else x
    )
    print(len(df['location'].unique()))  # 242 dimensions
    verify_data(df)

    df.to_pickle(os.path.join(OUT_FOLDER, 'feature_engineered.pkl'))


if __name__ == '__main__':
    main()
