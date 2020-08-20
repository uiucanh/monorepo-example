import numpy as np


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def r2(y_test: np.ndarray, y_pred: np.ndarray):
    y_mean = np.mean(y_test)
    ss_tot = np.square(y_test - y_mean).sum()
    ss_res = np.square(y_test - y_pred).sum()
    result = 1 - ss_res / ss_tot
    return result
