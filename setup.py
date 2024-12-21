from setuptools import setup, find_packages

setup(
    name="windsurf-ml-demo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "pyspark",
        "scikit-learn"
    ]
)
