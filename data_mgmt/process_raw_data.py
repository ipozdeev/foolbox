import pandas as pd
import numpy as np
import pickle

# Set the user contingent path
from foolbox.data_mgmt import set_credentials as set_cred
path = set_cred.gdrive_path("research_data/fx_and_events/")

# File to parse
file_name = "fx_and_macro_data_upd.xlsm"

# File names for pickling, see sample definitions below
name_dev_d = path + "data_dev_d.p"  # developed sample, daily data
name_dev_m = path + "data_dev_m.p"  # developed sample, monthly data
name_all_d = path + "data_all_d.p"  # all sample, daily data
name_all_m = path + "data_all_m.p"  # all sample, monthly data
name_all_evt = path + "events.p"  # events
name_all_on = path + "overnight_rates.p"

# Definitions and settings
eur_nan_date = "1999-01-04"           # First consistent EUR data begins on 5th
dem_nan_date = "1999-01-01"           # DEM is unavailable since...

# NZD and FIM daily observations where unavailable until...
msci_nzd_drop_until = "1986-12-31"
msci_fim_drop_until = "1986-12-31"

# Developed sample (by currency code)
developed = ['GBP', 'CHF', 'JPY', 'CAD', 'SEK', 'NOK', 'DKK', 'EUR',
             'AUD', 'NZD']

# Preset list of countries to be dropped from  all countries
countries_to_drop = ["UAH", "BGN", "SIT", "HUF", "HRK", "EGP", "PTE", "CZK",
                     "NLG", "BEF", "ITL", "FRF", "FIM", "ATS", "IEP", "GRD",
                     "ESP", ]

"""
===============================================================================
=== PART I: PARSE AND ORGANIZE THE RAW DATA                                 ===
===============================================================================
"""

# Load sheets from excel into a dictionary of dataframes
raw_data = pd.read_excel(path + file_name, sheetname=None, index_col=0)


# Define data filtering rules
trim_dates = raw_data["trim_dates"]  # dataframe, specifying which currencies
                                     # and periods to drop due to pegs, CIP
                                     # violations and joining EMU
trim_dates.drop("DEM", axis=0, inplace=True)  # drop DEM (spliced with EUR)

# Get list of currencies pegged for the whole sample
drop_pegged = trim_dates[trim_dates["drop_completely"] == "yes"].index.tolist()

# Get dataframe of currencies pegged to EUR, specifying date of the peg
eur_peg = trim_dates["peg_to_eur"].dropna().to_frame().T

# Get dataframe of periods of CIP violations
cip_violations = trim_dates[["start_of_peg_or_cip_violation",
                             "end_of_peg_or_cip_violation"]].dropna().T

# Loop through the dictionary processing data by dataframe
for df_name in raw_data.keys():
    # Process FX data:
    if df_name[:2] == "FX":
        # Create a temporary df for notational convenience
        temp_df = raw_data[df_name]

        # Construct list of levels for multiindex df:
        # how quoted (e.g. U$ or £) + original id (CHF, NOK)
        levels = [temp_df.iloc[2]] + [temp_df.columns]

        # Create multiindex
        multi_index = pd.MultiIndex.from_arrays(levels)
        temp_df.columns = multi_index  # assign multiindex to data
        temp_df.drop(["CURRENCY", "GEOGN", "Name"], inplace=True)
                                       # drop rows of strings

        # Locate currencies quoted not per 1 USD and convert them
        non_usd = temp_df.columns.get_level_values(0) != 'U$'        # non-USD
        temp_df.iloc[:, non_usd] = temp_df.iloc[:, non_usd].pow(-1)  # convert

        # Drop multiindex quote  and convert index to DateTime
        temp_df.columns = temp_df.columns.droplevel()
        temp_df.index = pd.to_datetime(temp_df.index)

        # Float shenanigans: reassemble the dataframe
        temp_df = pd.DataFrame(data=np.asfarray(temp_df.values),
                               index=temp_df.index, columns=temp_df.columns,
                               dtype=np.float)

        # Drop DEM and EURO after and prior last and first dates respectively
        temp_df["DEM"][dem_nan_date:] = np.nan
        temp_df["EUR"][:eur_nan_date] = np.nan

        # Get index of DEM first observation, replace NaNs with last value
        dem_start_date = temp_df["DEM"].first_valid_index()
        temp_df["DEM"].fillna(method='ffill', inplace=True)  # propagate DEM

        # Get integer indices for splicing period
        iterator_start = temp_df.index.get_loc(eur_nan_date)
        iterator_end = temp_df.index.get_loc(dem_start_date)

        # Splice the EUR and DEM series
        for t in np.arange(iterator_start, iterator_end-1, -1):
            temp_df['EUR'].ix[t] = temp_df['EUR'].ix[t+1] / \
                temp_df['DEM'].ix[t+1] * temp_df['DEM'].ix[t]

        # Drop DEM
        temp_df.drop("DEM", axis=1, inplace=True)

        # Drop currencies succeeded by EUR, periods of CIP violations, and
        # pegged currencies:

        # First, drop complete pegs
        temp_df.drop(drop_pegged, axis=1, inplace=True)

        # Second, delete observations for currencies after adoption of EUR peg
        for currency in eur_peg.columns:
            peg_date = eur_peg[currency].ix["peg_to_eur"]  # get the peg date
            temp_df[currency][peg_date:] = np.nan          # drop observations

        # Third, similarly, delete observations for periods of CIP violations
        for currency in cip_violations.columns:
            drop_from = cip_violations[currency]\
                                        .ix["start_of_peg_or_cip_violation"]
            drop_until = cip_violations[currency]\
                                        .ix["end_of_peg_or_cip_violation"]
            temp_df[currency][drop_from:drop_until] = np.nan

        # Rewrite the raw dataframe with processed one
        raw_data[df_name] = temp_df.astype(dtype = np.float)

    # Process commodity data:
    if df_name[:4] == "COMM":
        # Splice RBOB and Unleaded gasoline
        temp_df = raw_data[df_name]  # get the data to work with

        # Get the first valid date for RBOB gasoline
        rbob_start_date = temp_df["OILRBNY"].first_valid_index()

        # Reindex unleaded gasoline's price level to match RBOB's first price
        temp_df["GASUREG"] = temp_df["GASUREG"] /\
                             temp_df["GASUREG"].ix[rbob_start_date] *\
                             temp_df["OILRBNY"].ix[rbob_start_date]

        # Merge the spliced series
        temp_df["OILRBNY"].ix[:rbob_start_date] =\
                                        temp_df["GASUREG"].ix[:rbob_start_date]

        # Drop unleaded gasoline
        temp_df.drop(["GASUREG"], axis=1, inplace=True)

        # Rewrite the raw dataframe with processed one
        raw_data[df_name] = temp_df.astype(dtype=np.float)

    # Process the stock data
    if df_name == "MSCI":
        # Get the data
        temp_df = raw_data[df_name]
        # Drop New Zealand and Finland where daily data are unavailable
        temp_df["NZD"][:msci_nzd_drop_until] = np.nan
        temp_df["FIM"][:msci_fim_drop_until] = np.nan

# TODO: discuss whether MSCI indices should be aligned with FX here
        # # Drop stock indices where currencies succeeded by EUR, had periods of
        # # CIP violations, and were pegged:
        #
        # # First, drop complete pegs
        # stocks_set = set(temp_df.columns.tolist())  # stock columns
        # fx_set = set(drop_pegged)  # fx set
        # stocks_to_drop = list(fx_set.intersection(stocks_set))  #find intersection
        # temp_df.drop(stocks_to_drop, axis=1, inplace=True)
        #
        # # Second, delete observations for currencies after adoption of EUR peg
        # for currency in eur_peg.columns:  # check if stock index is available
        #     if currency in temp_df.columns:
        #         peg_date = eur_peg[currency].ix["peg_to_eur"]  # get the peg date
        #         temp_df[currency][peg_date:] = np.nan          # drop observations
        #
        # # Third, similarly, delete observations for periods of CIP violations
        # for currency in cip_violations.columns:
        #     if currency in temp_df.columns:  # check if stock index is available
        #         drop_from = cip_violations[currency]\
        #                                     .ix["start_of_peg_or_cip_violation"]
        #         drop_until = cip_violations[currency]\
        #                                     .ix["end_of_peg_or_cip_violation"]
        #         temp_df[currency][drop_from:drop_until] = np.nan

        # Rewrite the raw dataframe with processed one
        raw_data[df_name] = temp_df.astype(dtype=np.float)


"""
===============================================================================
=== PART II: ORGANIZE DAILY DATA                                            ===
===============================================================================
"""
# As a dictionary
daily_data = dict()

# FX spots
daily_data["spot_mid"] = raw_data["FX_spot_mid"]
daily_data["spot_bid"] = raw_data["FX_spot_bid"]
daily_data["spot_ask"] = raw_data["FX_spot_ask"]

# FX forwards
daily_data["fwd_mid"] = raw_data["FX_forward_mid"]
daily_data["fwd_bid"] = raw_data["FX_forward_bid"]
daily_data["fwd_ask"] = raw_data["FX_forward_ask"]

# Spot returns and forward discounts, mind the minus for spot rets
daily_data["fwd_disc"] =\
    np.log(daily_data["fwd_mid"]/daily_data["spot_mid"])
daily_data["spot_ret"] =\
    -np.log(daily_data["spot_mid"]/daily_data["spot_mid"].shift(1))

# Commodity spots and the corresponding returns
daily_data["comm_spot"] = raw_data["COMM_spot"]
daily_data["comm_spot_ret"] =\
    np.log(daily_data["comm_spot"]/daily_data["comm_spot"].shift(1))

# Commodity futures and the corresponding returns
daily_data["comm_fut_idx"] = raw_data["COMM_cont_futures"]
daily_data["comm_fut_ret"] =\
    np.log(daily_data["comm_fut_idx"]/daily_data["comm_fut_idx"].shift(1))

# MSCI Indices and the corresponding returns
daily_data["msci_price"] = raw_data["MSCI"]
daily_data["msci_ret"] =\
    np.log(daily_data["msci_price"]/daily_data["msci_price"].shift(1))

# Filter out the hobo-countries
# Process the all data
for key in daily_data.keys():
    if key[:4] != "comm":
        # Drop currencies with number of stocks in their markets is less than 5
        daily_data[key].drop(countries_to_drop, axis=1, inplace=True)
        # Allign fx data with msci data by column
        daily_data[key], _ = daily_data[key].\
            align(daily_data["msci_price"], join="inner", axis=1)

# Align stocks to currencies
daily_data["msci_price"], _ = daily_data["msci_price"].\
    align(daily_data["spot_mid"], join="inner", axis=1)
daily_data["msci_ret"], _ = daily_data["msci_ret"].\
    align(daily_data["spot_mid"], join="inner", axis=1)


"""
===============================================================================
=== PART III: ORGANIZE MONTHLY DATA                                         ===
===============================================================================
"""
# Iterate over the daily dictionary, use resample with .last() or .sum()
# methods where appropriate
monthly_data = dict()
for key in daily_data.keys():
    # If log returns, resample with .sum()
    if "_ret" in key:
        monthly_data[key] = daily_data[key].resample("M").sum()
    # Take the last values for the level sereis
    else:
        monthly_data[key] = daily_data[key].resample("M").last()

# Add excess returns
"""
For each month t, foreign currency for unit of USD:
Spot return:      - [s(t) - s(t-1)], minus because currencies are per 1 USD
Excess return:    f(t-1) - s(t) = - [s(t) - s(t-1)] + [f(t-1) - s(t-1)]
Forward discount: f(t) - s(t)
"""
monthly_data["rx"] =\
    np.log(monthly_data["fwd_mid"].shift(1)/monthly_data["spot_mid"])


"""
===============================================================================
=== PART IV: ALLOCATE DATA INTO SAMPLES                                     ===
===============================================================================
"""
# Developed data are also in dictionaries
daily_data_developed = dict()
monthly_data_developed = dict()

# Populate the developed subsample
for key in monthly_data.keys():
    if key[:4] != "comm":
        monthly_data_developed[key] = monthly_data[key][developed]
    else:
        monthly_data_developed[key] = monthly_data[key]

# Same for daily frequency
for key in daily_data.keys():
    if key[:4] != "comm":
        daily_data_developed[key] = daily_data[key][developed]
    else:
        daily_data_developed[key] = daily_data[key]

"""
===============================================================================
=== PART V: EVENTS DATA                                                     ===
===============================================================================
"""
opec = pd.read_csv(path+"opec_meetings_1984_2016.txt", sep=',',
    index_col=0, parse_dates=True, header=0)
fomc = pd.read_csv(path+"fomc_meetings_1994_2017.txt", sep=',',
    index_col=0, parse_dates=True, header=0)*100
boe = pd.read_csv(path+"boe_meetings_1997_2017.txt", sep=',',
    index_col=0, parse_dates=True, header=0)*100
ecb = pd.read_csv(path+"ecb_meetings_1999_2017.txt", sep=',',
    index_col=0, parse_dates=True, header=0)
norges = pd.read_csv(path+"norges_bank_meetings_1993_2017.txt", sep=",",
                    index_col=0, parse_dates=True, header=0)

# Riksbank is sort of a weirdo, implementing policy a week after announcements
riks = pd.read_csv(path+"riksbank_meetings_1994_2017.txt", sep=",",
                   index_col=0, parse_dates=True, header=0)
# Make the announcement date the index
riks = riks.where(pd.notnull(riks["announcement date"])).dropna()
riks["effective date"] = riks.index
riks.index = riks["announcement date"]
riks.index = riks.index.to_datetime()
riks.drop(["announcement date"], axis=1)

rba = pd.read_csv(path+"rba_meetings_1990_2017.txt", sep=",",
                  index_col=0, parse_dates=True, header=0)
rbnz = pd.read_csv(path+"rbnz_meetings_1999_2017.txt", sep=",",
                   index_col=0, parse_dates=True, header=0)
boc = pd.read_csv(path+"boc_meetings_2001_2017.txt", sep=",",
                  index_col=0, parse_dates=True, header=0)
snb = pd.read_csv(path+"snb_meetings_2000_2017.txt", sep=",",
                  index_col=0, parse_dates=True, header=0)
boj = pd.read_csv(path+"boj_meetings_1998_2017.txt", sep=",",
                  index_col=0, parse_dates=True, header=0)
us_cpi = pd.read_csv(path+"cpi_releases_1994_2017.txt", sep=',',
                     index_col=0, parse_dates=True, header=0)

joint_events = pd.concat([rba.rate.diff(), boc.rate.diff(),
                          snb.ix[snb.scheduled].mid.diff(),
                          ecb.deposit.diff(), boe.rate.diff(),
                          boj.rate.diff(), norges.rate.diff(),
                          rbnz.rate.diff(), riks.rate.diff(),
                          fomc.rate.diff()],
                         join="outer", axis=1)
joint_events.columns = ["aud", "cad", "chf", "eur", "gbp", "jpy", "nok", "nzd",
                        "sek", "usd"]

joint_events_lvl = pd.concat(
    [rba.rate, boc.rate, snb.ix[snb.scheduled].mid, ecb.deposit,
     boe.rate, boj.rate, norges.rate, rbnz.rate, riks.rate, fomc.rate],
    join="outer", axis=1)

joint_events_lvl.columns = ["aud", "cad", "chf", "eur", "gbp", "jpy", "nok",
                            "nzd", "sek", "usd"]

events = {
    "opec": opec,
    "fomc": fomc,
    "boe": boe,
    "ecb": ecb,
    "norges": norges,
    "riks": riks,
    "rba": rba,
    "rbnz": rbnz,
    "boc": boc,
    "snb": snb,
    "boj": boj,
    "joint_cbs": joint_events,
    "joint_cbs_lvl": joint_events_lvl,
    "us_cpi": us_cpi}


"""
===============================================================================
=== PART VI: OVERNIGHT EFFECTIVE RATES                                      ===
===============================================================================
"""
# read in names from tab "iso"
on_names = pd.read_excel(path+"overnight_ref_rates_1994_2017_d.xlsx",
    sheetname="iso")
# read in bloomberg output
raw = pd.read_excel(path+"overnight_ref_rates_1994_2017_d.xlsx",
    sheetname="hardcopy", skiprows=1)
# disassemble into separate dataframes
raw_by_cur = [
    raw.ix[:,(p*3):(p*3+2)].dropna() for p in range(on_names.shape[1])]
# break each piece into index and data, concatenate all
on = pd.concat(
    [pd.Series(index=p.iloc[:,0].values, data=p.iloc[:,1].values) \
        for p in raw_by_cur],
    axis=1)
# add back columns, sort
on.columns = on_names.columns
on = on[sorted(on.columns)]

"""
===============================================================================
=== , PICKLE IT FOR THE GREAT JUSTICE                                       ===
===============================================================================
"""
# Set columns to lower case and sort alphabetically for every dataset
for key in daily_data.keys():
    daily_data[key].columns = [col.lower() for col in daily_data[key].columns]
    daily_data[key] = daily_data[key][sorted(daily_data[key].columns)]

for key in daily_data_developed.keys():
    daily_data_developed[key].columns = [col.lower() for col in
                                         daily_data_developed[key].columns]
    daily_data_developed[key] =\
        daily_data_developed[key][sorted(daily_data_developed[key].columns)]

for key in monthly_data.keys():
    monthly_data[key].columns = [col.lower() for col in
                                 monthly_data[key].columns]
    monthly_data[key] = monthly_data[key][sorted(monthly_data[key].columns)]

for key in monthly_data_developed.keys():
    monthly_data_developed[key].columns = [col.lower() for col in
                                           monthly_data_developed[key].columns]
    monthly_data_developed[key] =\
        monthly_data_developed[key][sorted(monthly_data_developed[key].columns)]


# Pickle it for future (ab)use
with open(name_all_d, "wb") as fname:
    pickle.dump(daily_data, fname)
with open(name_all_m, "wb") as fname:
    pickle.dump(monthly_data, fname)
with open(name_dev_d, "wb") as fname:
    pickle.dump(daily_data_developed, fname)
with open(name_dev_m, "wb") as fname:
    pickle.dump(monthly_data_developed, fname)
with open(name_all_evt, "wb") as fname:
    pickle.dump(events, fname)
with open(name_all_on, "wb") as fname:
    pickle.dump(on, fname)
