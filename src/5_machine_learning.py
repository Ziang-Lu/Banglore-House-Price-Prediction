#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5: Machine Learning (Model Building)
"""

__author__ = 'Ziang Lu'

import json
import os

from pandas import DataFrame, Series
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (
    GridSearchCV, ShuffleSplit, cross_val_score, train_test_split
)

from utils import OUT_FOLDER, drop_columns, read_from_pickle, save_to_pickle


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


def main():
    df = read_from_pickle('onehot_encoded.pkl')

    X = drop_columns(df, ['price'])
    y = df['price']

    # Normal linear regression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    # Training
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    # Predict
    print(lr_model.score(X_test, y_test))

    # k-fold validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    print(cross_val_score(LinearRegression(), X, y, cv=cv))

    # Try some algorithms, and find the best one using GridSearchCV

    print(_find_best_algo_using_gridsearchcv(X, y))

    # Dump the necessary data
    with open(os.path.join(OUT_FOLDER, 'columns.json'), 'w') as f:
        columns = list(map(lambda x: x.strip().lower(), X.columns))
        json.dump(columns, f)

    # According to the result, linear regression model without normalization
    # performs the best.
    save_to_pickle(lr_model, 'linear_regression_model.sav')


if __name__ == '__main__':
    main()
