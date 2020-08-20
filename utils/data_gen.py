import numpy as np


def generate_linear_data(n_samples: int = 100, n_features: int = 1,
                         x_min: int = -5, x_max: int = 5,
                         m_min: int = -10, m_max: int = 10,
                         noise_strength: int = 1, seed: int = None,
                         bias: int = 10):
    # Set the random seed
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(x_min, x_max, size=(n_samples, n_features))
    m = np.random.uniform(m_min, m_max, size=n_features)
    y = np.dot(X, m).reshape((n_samples, 1))

    if bias != 0:
        y += bias

    # Add Gaussian noise
    y += np.random.normal(size=y.shape) * noise_strength

    return X, y


def split_dataset(X: np.ndarray, y: np.ndarray,
                  test_size: float = 0.2, seed: int = 0):
    # Set the random seed
    np.random.seed(seed)

    # Shuffle dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Splitting
    X_split_point = int(len(X) * (1 - test_size))
    y_split_point = int(len(y) * (1 - test_size))
    X_train, X_test = X[:X_split_point], X[X_split_point:]
    y_train, y_test = y[:y_split_point], y[y_split_point:]

    return X_train, X_test, y_train, y_test
