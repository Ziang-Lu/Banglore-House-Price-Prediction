import numpy as np
import pandas as pd

from utils import drop_columns, verify_data


def main():
    df = pd.read_pickle('feature_engineered.pkl')

    # Add another temporary feature which is useful
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    verify_data(df)

    # For each location, we check the "price_per_sqft" column.
    # Define outliers to be those out of the 1 standard deviation
    df_out = pd.DataFrame()
    for _, subdf in df.groupby('location'):
        pps_col = subdf['price_per_sqft']
        if len(pps_col) >= 5:
            mean = np.mean(pps_col)
            std = np.std(pps_col)
            reduced_df = subdf[
                (pps_col >= (mean - std)) & (pps_col <= (mean + std))
            ]
        else:
            reduced_df = subdf
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    df = df_out
    verify_data(df)

    # Check "bedrooms" and "bathrooms" columns
    # Define outliers to be those with # of bedrooms < # of bathrooms
    df = df[df['bedrooms'] >= df['bathrooms']]
    verify_data(df)

    # Drop the temporary feature
    df = drop_columns(df, ['price_per_sqft'])
    verify_data(df)

    df.to_pickle('outlier_removed.pkl')


if __name__ == '__main__':
    main()
