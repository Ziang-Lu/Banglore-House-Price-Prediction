import os

import numpy as np
import pandas as pd

from utils import OUT_FOLDER, drop_columns, drop_indices, verify_data


def main():
    df = pd.read_pickle(os.path.join(OUT_FOLDER, 'feature_engineered.pkl'))

    # Add another temporary feature which is useful
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    verify_data(df)

    # Check the "price_per_sqft" column for each location
    # Naturally, we think that for the same location, the "price_per_sqft"
    # should not vary too much.
    # => Define outliers to be those out of the 1 standard deviation
    df_out = pd.DataFrame()
    for _, subdf in df.groupby('location'):
        loc_pps = subdf['price_per_sqft']
        mean = np.mean(loc_pps)
        std = np.std(loc_pps)
        reduced_df = subdf[
            (loc_pps > (mean - std)) & (loc_pps < (mean + std))
        ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    df = df_out
    verify_data(df)

    # Check the "bedrooms" and "price_per_sqft" columns for each location
    # Naturally, we think that for the same location, the "price_per_sqft" with
    # N bedrooms should be >= that with (N - 1) bedrooms.
    # => Define outliers to be those with N bedrooms and has "price_per_sqft" <=
    #    the mean of those with (N - 1) bedrooms
    exclude_indices = np.array([])
    for _, loc_subdf in df.groupby('location'):
        # Collect the statistics for the current location
        bedroom_stats = {}
        for n_bedroom, loc_bedroom_subdf in loc_subdf.groupby('bedrooms'):
            # Collect the statistics for the current location with n bedrooms
            bedroom_stats[n_bedroom] = {
                'mean': np.mean(loc_bedroom_subdf['price_per_sqft']),
                'count': loc_bedroom_subdf.shape[0]
            }

        for n_bedroom, loc_bedroom_subdf in loc_subdf.groupby('bedrooms'):
            pre_stats = bedroom_stats.get(n_bedroom - 1, {})
            if pre_stats and pre_stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    loc_bedroom_subdf[
                        loc_bedroom_subdf['price_per_sqft'] <= pre_stats['mean']
                    ].index.values
                )
    df = drop_indices(df, exclude_indices)
    verify_data(df)

    # Compare "bedrooms" and "bathrooms" columns
    # => Define outliers to be those with # of bedrooms < # of bathrooms
    df = df[df['bedrooms'] >= df['bathrooms']]
    verify_data(df)

    # Drop the temporary feature
    df = drop_columns(df, ['price_per_sqft'])
    verify_data(df)

    df.to_pickle(os.path.join(OUT_FOLDER, 'outlier_removed.pkl'))


if __name__ == '__main__':
    main()
