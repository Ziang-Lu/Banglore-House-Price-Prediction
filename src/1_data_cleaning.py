import os

import pandas as pd

from utils import DATA_FOLDER, OUT_FOLDER, verify_data, drop_columns


def _convert_to_sqft(expr: str) -> float:
    """
    Converts the given expression to a float number in square feet.
    :param expr: str
    :return: float
    """
    unit_to_sqft = {
        'Sq.Meter': 10.7639,
        'Perch': 272.25
    }

    if '-' in expr:
        parts = expr.split(' - ')
        lower, upper = float(parts[0]), float(parts[1])
        return (lower + upper) / 2
    elif not expr.isdigit():
        for unit in ['Sq.Meter', 'Perch']:
            if expr.endswith(unit):
                amount = float(expr[:-len(unit)])
                return amount * unit_to_sqft[unit]
    else:
        return float(expr)


def main():
    df1 = pd.read_csv(
        os.path.join(DATA_FOLDER, 'datasets_20710_26737_Bengaluru_House_Data')
    )
    verify_data(df1)

    # Remove unnecessary columns
    print(df1.groupby('area_type')['area_type'].agg('count'))
    df2 = drop_columns(
        df1, ['area_type', 'availability', 'society', 'balcony'])
    verify_data(df2)

    # For remaining columns, drop null values
    print(df2.isnull().sum())
    df3 = df2.dropna()
    verify_data(df3)

    # Clean the "location" column
    df3['location'].apply(lambda x: x.strip())

    # Uniform the "total_sqft" column data format
    # print(df3['total_sqft'].unique())
    df3['total_sqft'] = df3['total_sqft'].apply(_convert_to_sqft)
    verify_data(df3)

    # Uniform the "size" column data format
    # print(df4['size'].unique())
    df3['bedrooms'] = df3['size'].apply(lambda x: int(x.split()[0]))
    df4 = drop_columns(df3, ['size'])

    # Uniform the "bath" column
    # print(df4['bath'].unique())
    df4['bathrooms'] = df4['bath'].apply(lambda x: int(x))
    df5 = drop_columns(df4, ['bath'])
    verify_data(df5)

    df5.to_pickle(os.path.join(OUT_FOLDER, 'data_cleaned.pkl'))


if __name__ == '__main__':
    main()
