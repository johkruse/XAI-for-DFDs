# Explainable Machine Learning for analysing deterministic grid frequency deviations

[![DOI](https://zenodo.org/badge/404294676.svg)](https://zenodo.org/badge/latestdoi/404294676)

Code accompanying the mansucript "Exploring deterministic frequency deviations with explainable AI".
Preprint: <https://arxiv.org/abs/2106.09538>


## Install

The code is written in Python (tested with python 3.7). To install the required dependencies execute the following commands:

```[python]
python3.7 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Usage

The `scripts` folder contains scripts to create the paper results and `notebooks` contains a notebook to reproduce the paper figures. The `scripts` contain a pipeline of 2 different stages:

* `1_train_test_split.py`: Split data set into train and test set and save data in a version folder.
* `2_model_fit.py`: Fit the XGBoost model, optimize hyper-parameters and calculate SHAP values.

## Data to run the code

We use the input data (features) and output data (targets) from [this zenodo repository](https://zenodo.org/record/5118352). To run the code, copy `input_actual.h5`,`input_forecast.h5` and `outputs.h5` into a folder `./data/CE/` in the repository folder. 

