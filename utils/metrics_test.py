import numpy as np
from utils.metrics import mean_absolute_percentage_error, r2


def test_working_mean_absolute_percentage_error():
    a, b = np.array([1, 1, 2, 2]), np.array([2, 2, 1, 1])
    result = mean_absolute_percentage_error(a, b)

    assert result == 75


def test_working_r2():
    a, b = np.array([1, 1, 2, 2]), np.array([2, 2, 1, 1])
    result = r2(a, b)

    assert result == -3
