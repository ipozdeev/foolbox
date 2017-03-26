from foolbox.api import *

# File to parse
file_name = "fx_wmr_data.xlsm"

# File names for pickling, see sample definitions below
name_dev_d = data_path + "data_wmr_dev_d.p"  # developed sample, daily data
name_dev_m = data_path + "data_wmr_dev_m.p"  # developed sample, monthly data

# Definitions and settings
eur_nan_date = "1999-01-04"           # First consistent EUR data begins on 5th
dem_nan_date = "1999-01-01"           # DEM is unavailable since...

"""
===============================================================================
=== PART I: PARSE AND ORGANIZE THE RAW DATA                                 ===
===============================================================================
"""

# Load sheets from excel into a dictionary of dataframes
raw_data = pd.read_excel(data_path + file_name, sheetname=None, index_col=0)

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

# FX weekly forwards
daily_data["fwd_mid_w"] = raw_data["FX_forward_mid_w"]
daily_data["fwd_bid_w"] = raw_data["FX_forward_bid_w"]
daily_data["fwd_ask_w"] = raw_data["FX_forward_ask_w"]


# Spot returns and forward discounts, mind the minus for spot rets
daily_data["fwd_disc"] =\
    np.log(daily_data["fwd_mid"]/daily_data["spot_mid"])
daily_data["spot_ret"] =\
    -np.log(daily_data["spot_mid"]/daily_data["spot_mid"].shift(1))

# Stubbornness corner rolling 5-day rx
daily_data["rx_5d"] =\
    np.log(daily_data["fwd_mid_w"].shift(5)/daily_data["spot_mid"])


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
For each month t, from the US investor's perspective:
Spot return:      - [s(t) - s(t-1)], minus because currencies are per 1 USD
Excess return:    f(t-1) - s(t) = - [s(t) - s(t-1)] + [f(t-1) - s(t-1)]
Forward discount: f(t) - s(t)
"""
monthly_data["rx"] =\
    np.log(monthly_data["fwd_mid"].shift(1)/monthly_data["spot_mid"])


"""
===============================================================================
=== , PICKLE IT FOR THE GREAT JUSTICE                                       ===
===============================================================================
"""
# Set columns to lower case and sort alphabetically for every dataset
for key in daily_data.keys():
    daily_data[key].columns = [col.lower() for col in daily_data[key].columns]
    daily_data[key] = daily_data[key][sorted(daily_data[key].columns)]

for key in monthly_data.keys():
    monthly_data[key].columns = [col.lower() for col in
                                 monthly_data[key].columns]
    monthly_data[key] = monthly_data[key][sorted(monthly_data[key].columns)]

# Pickle it for future (ab)use
with open(name_dev_d, "wb") as fname:
    pickle.dump(daily_data, fname)
with open(name_dev_m, "wb") as fname:
    pickle.dump(monthly_data, fname)
