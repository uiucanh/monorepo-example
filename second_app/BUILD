python_library(
    name = "second_app",
    sources = [
        "app.py"
    ],
    dependencies = [
        "first_app",
        "utils",
    ],
)

python_tests(
    name = 'app_test',
    sources = [
        "app_test.py"
    ],
    dependencies = [
        ":second_app",
    ]
)
