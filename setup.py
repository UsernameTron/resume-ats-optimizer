from setuptools import setup, find_packages

setup(
    name="enhanced-ats-optimizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "nltk",
        "pandas",
        "pytest",
        "pytest-asyncio"
    ]
)