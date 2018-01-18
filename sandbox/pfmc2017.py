import pandas as pd
from wp_ois.wp_settings import *
from matplotlib import dates as mdates

if __name__ == "__main__":
    path = "C:\\Users\\borisenko\\GoogleDrive\\discussions\\pfmc2017\\"
    file = "smi_data.xlsm"

    raw_data = pd.read_excel(path + file,
                             sheetname=["novartis", "nestle", "msci", "fx"],
                             index_col=0, parse_dates=True)

    s_dt = "2014-01-01"
    e_dt = "2017-01-31"

    from foolbox.api import *
    fx_by_tz = pd.read_pickle(data_path+"fx_by_tz_aligned_d.p")

    stocks_name = "nestle"
    fx_name = "eurchf"
    base_currency = fx_name[:3]

    this_title = "MSCI Switzerland in CHF and USD"

    label_spot = base_currency.upper() + "/CHF"
    label_stock_chf = "MSCI SWI in CHF"
    label_stock_usd = "MSCI SWI in USD"

    label_stock_chf = "Novarits SIX"
    label_stock_usd = "Novartis NYSE"

    label_stock_chf = "Nestle SIX"
    label_stock_usd = "Nestle XETRA"



    # Plot the results---------------------------------------------------------
    stock_data = raw_data[stocks_name].loc[s_dt:e_dt, :].ffill()
    spot = raw_data["fx"].loc[s_dt:e_dt, [fx_name]].ffill()

    # Rebase to 100
    stock_data = stock_data.div(stock_data.loc["2015-01-14", :], axis=1) * 100

    figure, ax = plt.subplots(2, sharex=True, figsize=(9, 6))

    ax_stocks = ax[0]
    ax_spot = ax[1]

    spot.squeeze().plot(ax=ax_spot, x_compat=True, color="k", linewidth=1.5,
                        label=label_spot)

    stock_data["chf"].plot(ax=ax_stocks, x_compat=True, color=new_red,
                           linewidth=1.5, label=label_stock_chf)

    stock_data[base_currency].plot(ax=ax_stocks, x_compat=True, color=new_blue,
                                   linewidth=1.5, label=label_stock_usd)

    ax_spot.legend(loc="lower center", fontsize=14, frameon=False)
    ax_stocks.legend(loc="upper left", fontsize=14, frameon=False)

    plt.setp(ax_spot.get_xticklabels(), rotation=0, ha="center")

    # OLOLO
    ax_spot.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
    ax_spot.xaxis.set_minor_locator(mdates.MonthLocator(range(12)))

    figure.text(0.0, 0.25, base_currency.upper() + "/CHF spot",
                va='center', rotation='vertical', fontsize=14)

    figure.text(0.0, 0.75, "Price Index: 2015-01-14 = 100",
                va='center', rotation='vertical', fontsize=14)

    # Save the figure
    figure.tight_layout(h_pad=0.5, pad=1.05)

    plt.savefig(path + "fig_" + stocks_name + ".pdf")




