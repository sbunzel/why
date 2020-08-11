import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from why import Explainer


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=0,
        random_state=42,
    )
    feat_names = [f"feat_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feat_names).assign(target=y)


@pytest.fixture
def sample_train(sample_data):
    n_train = int(0.8 * sample_data.shape[0])
    return sample_data[:n_train]


@pytest.fixture
def sample_test(sample_data):
    n_train = int(0.8 * sample_data.shape[0])
    return sample_data[n_train:]


@pytest.fixture
def sample_model():
    return RandomForestClassifier(n_estimators=20)


@pytest.fixture
def sample_explainer(sample_train, sample_test, sample_model):
    return Explainer(
        train=sample_train,
        test=sample_test,
        target="target",
        model=sample_model,
        features=None,
        mode="binary_classification",
        random_feature=False,
    )
