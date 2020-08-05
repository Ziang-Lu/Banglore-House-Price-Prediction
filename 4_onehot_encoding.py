import pandas as pd

from utils import drop_columns, verify_data


def main():
    df = pd.read_pickle('outlier_removed.pkl')

    # * ONE-HOT ENCODING *

    # Get one-hot encoding of the "location" column
    onehot_encoding = pd.get_dummies(df['location'])

    # Note:
    # To represent some location, we can use all 0s on other locations to
    # indicate that location, so we can safely drop one column of one-hot
    # encoding.
    onehot_encoding = drop_columns(onehot_encoding, ['others'])

    # Concatenate the one-hot encoding dataframe with the original dataframe
    df = pd.concat([df, onehot_encoding], axis='columns')
    # Safely drop the original "location" column
    df = drop_columns(df, ['location'])
    verify_data(df)

    df.to_pickle('onehot_encoded.pkl')


if __name__ == '__main__':
    main()
