#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: One-hot Encoding
"""

__author__ = 'Ziang Lu'

import pandas as pd

from utils import drop_columns, read_from_pickle, save_to_pickle, verify_data


def main():
    df = read_from_pickle('outlier_removed.pkl')

    # * ONE-HOT ENCODING *

    # Get one-hot encoding of the "location" column
    onehot_encoding = pd.get_dummies(df['location'])

    # Note:
    # To represent some location, we can use all 0s on other locations to
    # indicate that location, so we can safely drop one column of one-hot
    # encoding.
    onehot_encoding = drop_columns(onehot_encoding, ['Others'])

    # Concatenate the one-hot encoding dataframe with the original dataframe
    df = pd.concat([df, onehot_encoding], axis='columns')
    # Safely drop the original "location" column
    df = drop_columns(df, ['location'])
    verify_data(df)

    save_to_pickle(df, 'onehot_encoded.pkl')


if __name__ == '__main__':
    main()
