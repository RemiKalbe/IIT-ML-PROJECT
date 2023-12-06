from setuptools import setup, find_packages

setup(
    name="graph",
    version="0.1",
    packages=find_packages(include=["graph"]),
    install_requires=["polars", "pandas", "numpy"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
)
