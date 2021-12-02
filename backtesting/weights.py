import numpy as np
import pandas as pd


def rescale_weights(weights, leverage="net"):
    """Rescale weights of (possibly) leveraged positions.

    Parameters
    ----------
    weights : pandas.Series or pandas.DataFrame
        of position flags in form of +1, -1 and 0
    leverage : str
        'zero' to make position weights sum up to zero;
        'net' to make absolute weights of short and long positions sum up
            to one each (sum(abs(negative positions)) = 1);
        'unlimited' for unlimited leverage.
    """
    if isinstance(weights, pd.DataFrame):
        return weights.apply(rescale_weights, leverage=leverage)

    # weights are understood to be fractions of portfolio value
    if leverage.startswith("zero"):
        # NB: dangerous, because meaningless whenever there are long/sort
        #   positions only
        # make sure that pandas does not sum all nan's to zero
        row_lev = np.abs(weights).sum()

        # divide by leverage
        pos_weights = weights.divide(row_lev)

    elif leverage.startswith("net"):
        # deleverage positive and negative positions separately
        row_lev_pos = weights.where(weights > 0).sum()
        row_lev_neg = -1 * weights.where(weights < 0).sum()

        # divide by leverage
        pos_weights = \
            weights.where(weights < 0).divide(row_lev_neg)\
                .fillna(weights.where(weights > 0).divide(row_lev_pos))

    elif leverage.startswith("unlim"):
        pos_weights = np.sign(weights)

    else:
        raise NotImplementedError("Leverage not known!")

    return pos_weights


def position_flags_to_weights(position_flags, leverage="net"):
    """
    leverage : str
        "unlimited" for allowing unlimited leverage
        "net" for restricting it to 1
        "zero" for restricting that short and long positions net out to
            no-leverage
    """
    # position flags need be of a suitable type
    position_flags = position_flags.astype(float)

    res = rescale_weights(position_flags, leverage=leverage)

    return res
