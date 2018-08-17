from foolbox.api import *
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.utils import *

# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
start_date = pd.to_datetime(settings["sample_start"])
end_date = pd.to_datetime(settings["sample_end"])

# Set maturity of the OIS spread, i.e. '3m' means 3 month over 1 month OIS rate
test_maturity = "3M"
test_currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek']
#test_currencies = ["usd"]

# Load OIS and FX data
# with open(data_path + "ois_bloomberg.p", mode="rb") as halupa:
#     ois_data = pickle.load(halupa)

ois_data = pd.read_pickle(data_path+"ois_bloomi_1w_30y.p")

# with open(data_path+"daily_rx.p", mode="rb") as fname:
#     data_rx = pickle.load(fname)

data_rx = pd.read_pickle(data_path+"daily_rx.p")

# with open(data_path+"data_dev_d.p", mode="rb") as fname:
#     stock_data = pickle.load(fname)["msci_ret"][test_currencies]

stock_data = pd.read_pickle(data_path+"data_dev_d.p")

rx = data_rx["rx"]
rx["usd"] = rx.mean(axis=1)
spot = data_rx["spot"]

# Cosntruct OIS spreads
ois_spreads = dict()
for key in ois_data.keys():
    ois_spreads[key] = \
        ois_data[key].sub(ois_data[key]["ON"], axis=0).drop(["ON"], axis=1) \
    [start_date:end_date]

ois_spreads = align_and_fillna(ois_spreads, "B", method="ffill")

# Construct rx and corresponding spreads dataframes
test_rx = rx[test_currencies]
test_spot = spot[test_currencies]
stock_in_usd = stock_data + test_spot
stock_hedged = stock_data - (test_rx-test_spot)
test_spreads = pd.DataFrame(index=test_rx.index, columns=test_rx.columns)
spread_usd = ois_spreads["usd"][test_maturity]
for curr in test_rx.columns:
    test_spreads[curr] = ois_spreads[curr][test_maturity]#-spread_usd

st = multiple_timing(test_rx, test_spreads.rolling(5).mean().shift(5),
                     xs_avg=True)

st.cumsum().plot()

print(taf.descriptives(st.replace(0, np.nan)*100, 261))

signal = rx.rolling(261).corr(stock_data).resample("M").last().shift(1)
lol = poco.rank_sort_adv(test_rx.resample("M").sum(),
                         signal, 3)
lol = poco.get_factor_portfolios(lol, hml=True)
lol.cumsum().plot()
taf.descriptives(lol*100, 12)


wut = pd.DataFrame(index=test_rx.index, columns=test_rx.columns)
for curr in test_rx.columns:
    wut[curr] = ois_data[curr][test_maturity] -\
                ois_data["usd"][test_maturity]

st = multiple_timing(test_rx, test_spreads.rolling(22).mean().shift(5),
                     xs_avg=True)
