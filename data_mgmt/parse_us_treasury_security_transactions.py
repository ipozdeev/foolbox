"""Parses US Treasury data on transctions in stocks and bonds.
'in' and 'out' correspond to gross purchases and sales by foreigners from and
to US residents repsctively. That said, thosa are capital in and outflows
"""

import pandas as pd
import numpy as np
import pickle
from foolbox.data_mgmt import set_credentials as set_cred
from pandas.tseries.offsets import MonthEnd

if __name__ == "__main__":
    # Set the user contingent path
    path = set_cred.gdrive_path("research_data/fx_and_events/")

    # Set the i/o names
    file_name = "us_treasury_security_transactions.csv"
    out_name = "tic_flows_bonds_stocks.p"

    col_names = ["country", "country_code", "date",

                 "in_tbonds", "in_gov_corp_bonds", "in_us_corp_bonds",
                 "in_us_stocks", "in_foreign_bonds", "in_foreign_stocks",

                 "out_tbonds", "out_gov_corp_bonds", "out_us_corp_bonds",
                 "out_us_stocks", "out_foreign_bonds", "out_foreign_stocks"]

    # Columns with numbers
    float_cols = col_names[3:]

    # Core Eurozone countries (adopted EUR since 1999)
    eurozone_core = ["Austria", "Belgium", "Finland", "France", "Germany",
                     "Ireland", "Italy", "Luxembourg", "Netherlands",
                     "Portugal", "Spain"]

    # Countries to report
    report_countries = {"aud": ["Australia"], "cad": ["Canada"],
                        "chf": ["Switzerland"], "dkk": ["Denmark"],
                        "eur": eurozone_core, "gbp": ["United Kingdom"],
                        "jpy": ["Japan"], "nzd": ["New Zealand"],
                        "nok": ["Norway"], "sek": ["Sweden"]}
    currency_order = ["aud", "cad", "chf", "dkk", "eur", "gbp", "jpy", "nzd",
                      "nok", "sek"]

    # Read the file
    raw_data = pd.read_csv(path+file_name, skiprows=19, names=col_names,
                           thousands=",", quotechar="\"",
                           index_col=["date"], parse_dates=True,
                           date_parser=lambda x: pd.Timestamp(x) + MonthEnd())

    # Convert to floats
    raw_data.drop(["country_code"], axis=1, inplace=True)
    raw_data[float_cols] = raw_data[float_cols].astype(np.float64)

    # Reset index for easier groupby
    raw_data.reset_index(inplace=True)

    # Aggregate by currency for each date
    by_curr = dict()
    for curr, countries in report_countries.items():
        by_curr[curr] = raw_data.loc[raw_data["country"].isin(
            countries)].groupby("date").sum()

    # Aggregate by data type
    by_dtype = dict()
    for dtype in float_cols:
        this_df = pd.DataFrame(columns=currency_order)
        for curr, df in by_curr.items():
            this_df.loc[:, curr] = df[dtype]

        by_dtype[dtype] = this_df

    # Save the data
    with open(path + out_name, mode="wb") as halupa:
        pickle.dump(by_dtype, halupa)
