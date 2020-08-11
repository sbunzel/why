import pandas as pd
import pytest
from why import Explainer


def test_explainer_provides_required_attributes(sample_explainer: Explainer) -> None:
    attributes = [
        "train",
        "test",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "model",
        "feature_names",
        "target",
        "train_preds",
        "test_preds",
    ]
    for attr in attributes:
        getattr(sample_explainer, attr)


def test_insert_random_num_adds_new_numeric_column(sample_explainer: Explainer) -> None:
    train = sample_explainer.train
    train_with_random = sample_explainer.insert_random_num(train, copy=True)
    assert train_with_random.shape[1] == train.shape[1] + 1
    assert "RANDOM_NUM" in train_with_random.columns
    assert pd.api.types.is_numeric_dtype(train_with_random["RANDOM_NUM"])


def test_explainer_raises_incompatible_input_data(sample_model) -> None:
    train_incompatible_col_names = pd.DataFrame(
        {"feat_0": [1.0, 2.0, 3.0], "feat_1": [3.0, 4.0, 5.0], "target": [0, 0, 0]}
    )
    test_incompatible_col_names = pd.DataFrame(
        {"feat_0": [1.0, 2.0], "feat_2": [3.0, 4.0], "target": [1, 0]}
    )
    with pytest.raises(ValueError):
        Explainer(
            train_incompatible_col_names,
            test_incompatible_col_names,
            target="target",
            model=sample_model,
        )

    train_incompatible_dtypes = pd.DataFrame(
        {"feat_0": ["a", "b", "c"], "feat_1": [3.0, 4.0, 5.0], "target": [0, 0, 0]}
    )
    test_incompatible_dtypes = pd.DataFrame(
        {"feat_0": ["a", "b"], "feat_1": [3.0, 4.0], "target": [1, 0]}
    )
    with pytest.raises(ValueError):
        Explainer(
            train_incompatible_dtypes,
            test_incompatible_dtypes,
            target="target",
            model=sample_model,
        )
