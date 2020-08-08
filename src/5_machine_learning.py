import os
import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (
    GridSearchCV, ShuffleSplit, cross_val_score, train_test_split
)

from utils import OUT_FOLDER, drop_columns


def _find_best_algo_using_gridsearchcv(X: DataFrame, y: Series) -> DataFrame:
    """
    Finds the best algorithm using GridSearchCV.
    :param X: DataFrame
    :param y: Series
    :return: DataFrame
    """
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    scores = []
    for algo, config in algos.items():
        gs = GridSearchCV(
            config['model'],
            config['params'],
            cv=cv,
            return_train_score=False
        )
        gs.fit(X, y)
        scores.append({
            'model': algo,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return DataFrame(scores, columns=['model', 'best_score', 'best_params'])


def predict_score(X: DataFrame, lr_clf: LinearRegression, location: str,
                  total_sqft: float, bedrooms: int, bathrooms: int) -> float:
    x = np.zeroes(len(X.columns))
    x[0] = total_sqft
    x[1] = bedrooms
    x[2] = bathrooms
    loc_idx = np.where(X.columns == location)[0][0]
    if loc_idx >= 0:
        x[loc_idx] = 1
    return lr_clf.predict([x])[0]


def main():
    df = pd.read_pickle(os.path.join(OUT_FOLDER, 'onehot_encoded.pkl'))

    X = drop_columns(df, ['price'])
    y = df['price']

    # Normal linear regression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    # Training
    lr_clf = LinearRegression()
    lr_clf.fit(X_train, y_train)
    # Predict
    print(lr_clf.score(X_test, y_test))

    # k-fold validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    print(cross_val_score(LinearRegression(), X, y, cv=cv))

    # Try some algorithms, and find the best one using GridSearchCV

    print(_find_best_algo_using_gridsearchcv(X, y))
    # According to the result, linear regression model without normalization
    # performs the best.

    with open(os.path.join(OUT_FOLDER, 'linear_regression.sav'), 'wb') as f:
        pickle.dump(lr_clf, f)


if __name__ == '__main__':
    main()
