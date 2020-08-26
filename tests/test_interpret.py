import numpy as np
import pandas as pd
from why import Explainer
from why import interpret


def test_calculate_correlation_should_return_correct_dimensions(
    sample_explainer: Explainer,
):
    corr = interpret.FeatureCorrelation(sample_explainer).calculate_correlation().corr
    n_feats = len(sample_explainer.feature_names)
    assert corr.shape[0] == n_feats
    assert corr.shape[1] == n_feats


def test_maybe_sample_should_return_maximum_sample_size_rows():
    df = pd.DataFrame(data={"data": np.random.rand(100)})
    sample = interpret._maybe_sample(df, sample_size=50)
    assert sample.shape[0] <= 50
