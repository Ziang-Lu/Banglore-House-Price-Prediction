def verify_data(df):
    print(df.shape)
    print(df.head())


def drop_columns(df, columns):
    return df.drop(columns, axis='columns')
