#!usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6: Prediction
"""

__author__ = 'Ziang Lu'

import json
import os

import numpy as np

from utils import OUT_FOLDER, read_from_pickle


def predict(location: str, total_sqft: float, bedrooms: int,
            bathrooms: int) -> float:
    """
    Predicts the score for the given features.
    :param location: str
    :param total_sqft: float
    :param bedrooms: int
    :param bathrooms: int
    :return: float
    """
    with open(os.path.join(OUT_FOLDER, 'columns.json')) as f:
        columns = json.load(f)

    lr_model = read_from_pickle(
        'linear_regression_model.sav', is_dataframe=False)

    x = np.zeroes(len(columns))
    x[0] = total_sqft
    x[1] = bedrooms
    x[2] = bathrooms
    try:
        loc_idx = columns.index(location)
        x[loc_idx] = 1
    except ValueError:
        pass
    return lr_model.predict([x])[0]
