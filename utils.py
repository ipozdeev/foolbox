import pandas as pd

def align_and_fillna(data, reindex_freq=None, **kwargs):
    """
    Parameters
    ----------
    data : list-like
        of pandas.DataFrames or pandas.Series
    reindex_freq : str
        pandas frequency string, e.g. 'B' for business day
    kwargs : dict
        arguments to .fillna()
    """
    common_idx = pd.concat(data, axis=1, join="outer").index

    if reindex_freq is not None:
        common_idx = pd.date_range(common_idx[0], common_idx[-1],
            freq=reindex_freq)

    new_data = None

    if isinstance(data, dict):
        new_data = {}
        for k,v in data.items():
            new_data.update({k: v.reindex(index=common_idx).fillna(**kwargs)})
    elif isinstance(data, tuple):
        new_data = tuple(
            [p.reindex(index=common_idx).fillna(**kwargs) for p in data])
    elif isinstance(data, list):
        new_data = [p.reindex(index=common_idx).fillna(**kwargs) for p in data]

    return new_data
