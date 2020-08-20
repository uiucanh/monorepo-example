from utils.metrics import r2
from utils.data_gen import generate_linear_data, split_dataset
from first_app.app import Model
import sys
import os

# Enable import from outer directory
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/..")


def main():
    X, y = generate_linear_data()

    m = Model(X, y)
    m.split()
    m.fit()
    m.predict()

    result = r2(m.y_test, m.y_pred)
    print("R2:", result)
    return result


if __name__ == '__main__':
    _ = main()
