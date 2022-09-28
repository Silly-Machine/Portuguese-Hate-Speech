## Portuguese hate speech detection

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/FelipeRamosOliveira/Portfolio/pulls)
[![GitHub issues](https://img.shields.io/github/issues/FelipeRamosOliveira/Portfolio.svg)](https://img.shields.io/github/issues/FelipeRamosOliveira/Portfolio.svg)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-000000.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Scope

This project aims to develop a classification model capable of identifying the presence of hate speech in texts written in portuguese.The original dataset was built by [Fortuna](https://github.com/paulafortuna) *et. al.* and used in the article [Hierarchically Labeled Portuguese Hate Speech Dataset](https://aclanthology.org/W19-3510.pdf). Every attempt has been taken to protect the identity of the Twitter users in the dataset by modifying it appropriately. The dataset is strictly for research purposes and any attempt to violate the privacy of the Twitter users mentioned knowingly or unknowingly will not be liable to the authors of the paper or repository.

## Directory Structure

The project is organized in the following directory structure:

```sh
.
├── data            --> persisted data
├── notebooks       --> proofs of concept
├── pyproject.toml  --> configuration
└── src             --> auxilar code
```

## Setup

Clone the project:

```sh
git clone git@github.com:Silly-Machine/Twitter-Hate-Speech.git
```

Is highly recommended to create the following [`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual environment before installing the dependencies.

```sh
conda create -n hate-speech python=3.8
```

Activate the virtual environment:

```sh
conda activate hate-speech
```

In the virtual environment, install [`poetry`](https://python-poetry.org/) as the package manager :

```sh
conda install poetry
```

Install project dependencies:

```sh
poetry install
```
