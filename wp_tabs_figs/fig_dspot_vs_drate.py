from foolbox.api import *
import matplotlib.pyplot as plt
import seaborn as sns


path_to_data = set_credentials.set_path("research_data/fx_and_events/")


def plot_one(x, y, event_mask):
    """

    Parameters
    ----------
    x
    y
    event_mask
    kwargs

    Returns
    -------

    """
    # concat both
    both = pd.concat((y, x, event_mask), axis=1)

    # melt
    molten = both.melt(value_vars=y.columns, id_vars=[x.name, event_mask.name],
                       value_name="return", var_name="pre_or_post")

    # hue by quantile
    # hue_col = pd.cut(rate, [-5, -0.10, 0.10, 5]).rename("rate_bin")
    # both = pd.concat((both, hue_col), axis=1)

    ax = sns.lmplot(x=x.name, y="return",
                    data=molten,
                    fit_reg=True, col="pre_or_post",
                    hue="event_window", legend=False,
                    markers='.', sharey=True, size=5, aspect=4/3,
                    scatter_kws={"alpha": 0.66})

    ax.axes[0][0].set_ylim((-7.5, 7.5))

    plt.legend(loc='lower left', title="within event window")
    # ax.axes[0][0].invert_yaxis()

    return ax


if __name__ == '__main__':
    # fx
    fx_data = pd.read_pickle(path_to_data + "data_wmr_dev_d.p")
    rs = fx_data["spot_ret"] * 100

    # o/n rates
    rate_on = pd.read_pickle(path_to_data + "overnight_rates.p")

    # events
    events_data = pd.read_pickle(path_to_data + "events.p")
    events = events_data["joint_cbs"]

    # align
    rs, rate_on = rs.align(rate_on, axis=0, join="outer")
    rs, rate_on = rs.align(rate_on, axis=1, join="inner")

    # rolling spot returns
    rs_10d_pre = rs.rolling(10, min_periods=1).mean().shift(1) * 10
    rs_2d_post = rs.rolling(2, min_periods=1).mean().shift(-1) * 10

    # change in o/n rate
    d_rate_on = rate_on.diff()

    # windows outside events
    es = EventStudy(rs_10d_pre, events.loc[:, rs_10d_pre.columns],
                    window=(-10, -1, 0, 5))
    msk = ~es.mask_between_events

    # c = "aud"
    for c in rs.columns:
        ax = plot_one(
            x=d_rate_on[c].rename("rate"),
            y=pd.concat((rs_10d_pre[c].rename("pre_10d"),
                         rs_2d_post[c].rename("post_2d")), axis=1),
            event_mask=msk[c].rename("event_window"))

        ax.savefig(path_to_data + "output/dspot_vs_drate_{}.png".format(c))
