from foolbox.api import *

if __name__ == "__main__":
    # Parse this file
    meetings_file = data_path + \
        "central_bank_rates_raw_data/summary_all_central_banks.xlsx"

    # Parse these banks since these dates
    central_banks = {"rba": "2001-10", "boc": "2002-05", "snb": "2000-08",
                     "ecb": "2000-01", "boe": "2000-12", "rbnz": "2002-09",
                     "riks": "2004-08", "fomc": "2001-12"}
    # And unntil this date
    end_date = "2017-06-30"

    # Order cbs according to alphabetical order of currencies
    ordered_cbs = ["rba", "boc", "snb", "ecb", "boe", "rbnz", "riks", "fomc"]

    # Make output dataframes for both scheduled and unscheduled events
    out_all =\
        pd.DataFrame(index=ordered_cbs, columns=["total", "hikes", "cuts"])
    out_unscheduled =\
        pd.DataFrame(index=ordered_cbs, columns=["total", "hikes", "cuts"])

    # Read the data
    cb_data = pd.read_excel(meetings_file, sheetname=None, index_col=0)

    # Loop over cenral banks computing number of events
    for cb, start_date in central_banks.items():
        # Get the sample
        tmp_data = cb_data[cb].drop(["comments"], axis=1)[start_date:end_date]

        # Total number of events
        out_all.loc[cb, "total"] = tmp_data.change.count()
        out_unscheduled.loc[cb, "total"] = \
            tmp_data.loc[~tmp_data.scheduled].change.count()

        # Hikes
        out_all.loc[cb, "hikes"] = \
            tmp_data.loc[tmp_data.change > 0].change.count()
        out_unscheduled.loc[cb, "hikes"] = \
            tmp_data.loc[(tmp_data.change > 0) &
                         (~tmp_data.scheduled)].change.count()

        # Cuts
        out_all.loc[cb, "cuts"] = \
            tmp_data.loc[tmp_data.change < 0].change.count()
        out_unscheduled.loc[cb, "cuts"] = \
            tmp_data.loc[(tmp_data.change < 0) &
                         (~tmp_data.scheduled)].change.count()

    # Get the final table
    tab_cb_descr = \
        out_all.astype(str) + " (" + out_unscheduled.astype(str) + ")"
    print(tab_cb_descr)


