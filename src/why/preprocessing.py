import pandas as pd


__all__ = ["remove_high_corrs", "get_high_corrs"]


def remove_high_corrs(df: pd.DataFrame, min_corr: float = 1.0) -> pd.DataFrame:
    """Drop highly correlated features from a DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing the features
        min_corr (float, optional): Minimum value for Pearson correlation to consider a correlation as high. Defaults to 1.0. . Defaults to 1.0.

    Returns:
        pd.DataFrame: The feature data with highly correlated features removed
    """
    high_corrs = get_high_corrs(df, min_corr)
    return df.drop(columns=high_corrs["feature1"].unique())


def get_high_corrs(df: pd.DataFrame, min_corr: float = 1.0) -> pd.DataFrame:
    """Find features with high correlation

    Args:
        df (pd.DataFrame): DataFrame containing the features
        min_corr (float, optional): Minimum value for Pearson correlation to consider a correlation as high. Defaults to 1.0.

    Returns:
        pd.DataFrame: The highly correlated features
    """
    corrs = df.corr().reset_index()
    return (
        pd.melt(corrs, id_vars=["index"])
        .dropna()
        .rename({"index": "feature1", "variable": "feature2", "value": "corr"}, axis=1)
        .query(f"feature1 != feature2 and abs(corr) >= {min_corr}")
        .sort_values(["corr", "feature1", "feature2"])
        .loc[::2]
        .reset_index(drop=True)
    )
