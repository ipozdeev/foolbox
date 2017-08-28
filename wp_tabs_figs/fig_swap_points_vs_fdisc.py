import pandas as pd
from pandas.tseries.offsets import DateOffset
from matplotlib import dates as mdates, ticker
from foolbox.api import *
from foolbox.utils import remove_outliers
from foolbox.wp_tabs_figs.wp_settings import *
# %matplotlib

out_path = set_credentials.set_path("foresight_saga/tex/figs/", which="local")

# data ----------------------------------------------------------------------
# monthly
with open(data_path + "data_wmr_dev_m.p", mode='rb') as fname:
    fx_m = pickle.load(fname)

# forward discounts
fwd_disc_m = (np.exp(fx_m["fwd_disc"])-1)*100

# daily
with open(data_path + "fx_by_tz_aligned_d.p", mode='rb') as fname:
    fx_d = pickle.load(fname)

swap_ask = fx_d["tnswap_ask"]
swap_bid = fx_d["tnswap_bid"]

swap_ask = remove_outliers(swap_ask, 20).dropna(how="all")
swap_bid = remove_outliers(swap_bid, 20).dropna(how="all")

# mid quotes
fwd_disc_d = (swap_ask.ffill() + swap_bid.ffill())/2
# divide swap points through the spot price to arrive at forward discounts
fwd_disc_d = fwd_disc_d.divide(
    (fx_d["spot_bid"].ffill() + fx_d["spot_ask"].ffill())/2,
    axis=1)
# resample monthly, taking care of missing data, -1 is needed to convert to dr
fwd_disc_d_m = fwd_disc_d.resample('M').mean()*30*100*-1

# plot ----------------------------------------------------------------------
# subsample
curs = ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek']
s_dt = pd.to_datetime(settings["sample_start"])
e_dt = pd.to_datetime(settings["sample_end"])

fwd_disc_d_m = fwd_disc_d_m.loc[s_dt:e_dt,:]
fwd_disc_m = fwd_disc_m.loc[s_dt:e_dt,:]

# subplots
fig, ax = plt.subplots(len(curs), sharex=True, figsize=(8.27,10.0))

# set limits to the very last one (lowest) subplot, since the x-axis is shared
ax[-1].set_xlim((
    s_dt - DateOffset(months=3),
    e_dt + DateOffset(months=3)))

# set tick location and text appearance via Locator and DateFormatter:
#   major ticks are 2 years apart, minor - 1 year
ax[-1].xaxis.set_major_locator(mdates.YearLocator(2))
ax[-1].xaxis.set_minor_locator(mdates.YearLocator(1))
ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# plot 7 pieces
for p in range(len(curs)):
    # p = 0
    c = curs[p]

    # the two corresponding columns
    this_d_m = fwd_disc_d_m.loc[:,c]
    this_m = fwd_disc_m.shift(1).loc[:,c]

    # mean absolute difference
    mad = np.abs(this_d_m-this_m).mean()

    # plot
    this_d_m.plot(ax=ax[p], x_compat=True, color=new_blue, linewidth=1.5)
    this_m.plot(ax=ax[p], x_compat=True, color=new_red, linewidth=1.5)

    # limits
    ylim = (pd.concat((this_m, this_d_m), axis=0).dropna()\
        .quantile([0.05, 0.95])).values + np.array([-0.2, 0.2])
    ax[p].set_ylim(ylim)

    # aesthetics
    ax[p].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax[p].grid(axis='y', which="major", alpha=0.66)
    ax[p].grid(axis='x', which="major", alpha=0.66)
    ax[p].grid(axis='x', which="minor", alpha=0.33)
    ax[p].set_title(curs[p], y=0.99)

    # annotate with mad
    ax[p].annotate(r"$|err|={:3.2f}$".format(mad),
        xy=(0.1, 0.14), xycoords='axes fraction', backgroundcolor='w')

# x-axis tick settings
plt.setp(ax[-1].xaxis.get_majorticklabels(), rotation="horizontal",
    ha="center")
ax[-1].set_xlabel('', visible=False)

fig.text(0.0, 0.5, "return, in percent p.a.",
    va='center', rotation='vertical')

fig.tight_layout(h_pad=0.5, pad=1.05)

# save
fig.savefig(out_path + "daily_swap_pts_vs_monthly" + ".pdf")
