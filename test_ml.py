import pytest

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference

# Configuration
DATA_PATH = Path("data/census.csv")
LABEL = "salary"
CATS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# fixture
@pytest.fixture(scope="session")
def train_test():
    """Return small train/test DFs for fast testing"""
    df = pd.read_csv(DATA_PATH).sample(800, random_state=42)
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL]
    )
    return train, test


# test data
def test_process_data_shapes(train_test):
    """process_data returns espected X/y & fitted encoders."""
    train, _ = train_test
    X, y, enc, lb = process_data(
        train, categorical_features=CATS, label=LABEL, training=True
    )
    assert X.shape[0] == y.shape[0]
    assert enc is not None and lb is not None
    assert X.shape[1] > len(train.columns) - 1  # one-hot expansion


# test model training
def test_train_returns_fitted_model(train_test):
    """train_model returns a fitted AdaBoostClassifier."""
    train, _ = train_test
    X_train, y_train, _, _ = process_data(
        train, categorical_features=CATS, label=LABEL, training=True
    )
    model = train_model(X_train, y_train, use_gridsearch=False)
    assert isinstance(model, AdaBoostClassifier)
    # proof of fitting: AdaBoost exposes a populated estimators_ list
    assert hasattr(model, "estimators_")
    assert len(model.estimators_) > 0


# test model output
def test_inference_length_and_binary(train_test):
    """inference outputs one label per row and only 0/1."""
    train, test = train_test
    X_train, y_train, enc, lb = process_data(
        train, categorical_features=CATS, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CATS,
        label=LABEL,
        training=False,
        encoder=enc,
        lb=lb,
    )
    model = train_model(X_train, y_train, use_gridsearch=False)
    preds = inference(model, X_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)).issubset({0, 1})