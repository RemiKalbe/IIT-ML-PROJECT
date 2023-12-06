from setuptools import setup, find_packages

setup(
    name="localgnnanalyzer",
    version="0.1",
    packages=find_packages(include=["localgnnanalyzer", "localimpactcalculator"]),
    install_requires=[],
    setup_requires=["pytest-runner"],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
)
