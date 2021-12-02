import numpy as np
import pandas as pd


def estimate_covmat(data, assume_centered=False):
    """Estimate covariance matrix of a 2D array, possibly as E[xx].

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
    assume_centered : bool
        True to estimate as E[XX'], False to additionally subtract E[X]E[X]'

    Returns
    -------
    np.ndarray or pd.DataFrame
        depending on the input
    """
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(estimate_covmat(data.values, assume_centered),
                            index=data.columns, columns=data.columns)

    # E[XX']
    xx = data[:, :, np.newaxis] * \
        np.swapaxes(data[:, :, np.newaxis], 1, 2)

    exx = np.nanmean(xx, axis=0)

    if assume_centered:
        res = exx
    else:
        ex = np.nanmean(data, axis=0)
        exex = ex[np.newaxis, :] * ex[np.newaxis, :].T
        res = exx - exex

    return res
