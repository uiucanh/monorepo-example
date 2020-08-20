python_library(
    name = "first_app",
    sources = [
        "app.py"
    ],
    dependencies = [
        "//:numpy",
        "//:scikit-learn",
        "//:pytest",
        "utils",
    ],
)

python_tests(
    name = 'app_test',
    sources = [
        "app_test.py"
    ],
    dependencies = [
        ":first_app",
    ]
)