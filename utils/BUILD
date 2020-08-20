python_library(
    name = "utils",
    sources = [
        "data_gen.py",
        "metrics.py",
    ],
    dependencies = [
        "//:numpy"
    ]
)

python_tests(
    name = 'utils_test',
    sources = [
        "data_gen_test.py",
        "metrics_test.py",
    ],
    dependencies = [
        ":utils",
    ]
)
