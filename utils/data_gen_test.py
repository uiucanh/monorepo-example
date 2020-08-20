from utils.data_gen import generate_linear_data


def test_working_generate_linear_data():
    X, y = generate_linear_data(n_samples=10, n_features=2)

    assert X.shape == (10, 2)
    assert y.shape == (10, 1)
