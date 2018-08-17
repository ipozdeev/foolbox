import pandas as pd
from foolbox.data_mgmt import set_credentials as set_cred


if __name__ == "__main__":
    # Define the currency pairs of interest, data frequency and request period
    data_frequency = "m5"

    # Set output path and output names
    data_path = set_cred.gdrive_path("research_data/fx_and_events/")

    out_counter_usd_name = "fxcm_counter_usd_" + data_frequency + ".p"

    data = pd.read_pickle(data_path + out_counter_usd_name)

    import seaborn as sns
    import matplotlib.pyplot as plt
    colors = sns.color_palette("hls", 9)

    sns.set_palette(colors)
    to_plot = (data["ask_close"].loc["2016-04-20":"2016-05-05", ["aud"]].ffill() \
     .pct_change()+1).cumprod()

    fig, ax = plt.subplots()
    ax.plot(to_plot, label=to_plot.columns)
    ax.legend(list(to_plot.columns), ncol=3, loc="upper left", fontsize=18)


    df = data["ask_close"].reindex(freq="B")

