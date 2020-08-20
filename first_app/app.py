from utils.data_gen import generate_linear_data, split_dataset
from utils.metrics import mean_absolute_percentage_error, r2
from sklearn.linear_model import LinearRegression
import os
import sys

# Enable import from outer directory
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/..")


class Model:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = LinearRegression()
        self.y_pred = None

    def split(self, test_size=0.33, seed=0):
        self.X_train, self.X_test, self.y_train, self.y_test = split_dataset(
            self.X, self.y, test_size=test_size, seed=seed)

    def fit(self):
        self.m.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.m.predict(self.X_test)


def main():
    X, y = generate_linear_data()

    m = Model(X, y)
    m.split()
    m.fit()
    m.predict()

    print("MAPE:", mean_absolute_percentage_error(m.y_test, m.y_pred))


if __name__ == '__main__':
    main()
