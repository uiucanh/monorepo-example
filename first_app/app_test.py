import numpy as np
from first_app.app import Model


def test_model_working():
    X, y = np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1], [2]])

    m = Model(X, y)
    m.split()
    m.fit()
    m.predict()

    assert m.y_pred is not None
