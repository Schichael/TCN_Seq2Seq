from setuptools import setup
import os
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join("src", "tcn_sequence_models", "__init__.py"), "r") as fh:
    version = re.search('__version__ = "([^"]+)"', fh.read()).group(1)


with open("requirements/requirements.txt", "r") as f:
    required = f.read().splitlines()

with open("requirements/requirments_notebook.txt", "r") as f:
    required_notebooks = f.read().splitlines()

with open("requirements/requirements_dev.txt", "r") as f:
    required_dev = f.read().splitlines()

setup(
    name="tcn_sequence_models",
    version=version,
    author="Michael Schulten",
    author_email="michael.schulten@yahoo.de",
    url="https://github.com/Schichael/TCN_Seq2Seq",
    description="TCN based models for sequence forecasting and predictions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "tcn_sequence_models",
        "tcn_sequence_models.data_processing",
        "tcn_sequence_models.tf_models",
        "tcn_sequence_models.utils",
    ],
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=required,
    extras_require={"notebooks": required_notebooks, "dev": required_dev},
)
