"""
Calculate climatologies (1991–2020 reference) and monthly anomalies. Show the timeline of summer anomalies for Graz.
Which were the 5 hottest years? Mark them in the plot. Do the same for the three given parameters. Compute monthly anomalies as follows:
1. Compute the average, climatological monthly values for each month of the year, i.e. the mean January, mean February and so on.
Choose the climate normal period 1991-2020 to compute those means.
2. To compute the anomalies, you now need to subtract from each monthly value of the time series the corresponding mean, climatological value.

Plot the median, and the interquartile and interdecile range, for the mean, min, and max temperature of the whole time
period for each month, and include the current year, the year 2023 and 2024, and your birth year.

Quantify extreme heat: hot days (Tmax >= 30°C) and tropical nights (Tmin >= 20°C). Plot the yearly number
of hot days and tropical nights in a timeline.
"""

import pandas as pd
import matplotlib.pyplot as plt


def get_climate_data(csv_loc: str) -> pd.DataFrame:
    climate_data = pd.read_csv(csv_loc)
    climate_data["time"] = pd.to_datetime(climate_data.time)
    climate_data = climate_data.set_index("time")
    return climate_data


def get_monthly_aggregates(dataframe: pd.DataFrame) -> pd.DataFrame:
    clim = dataframe.groupby(dataframe.index.month).mean()
    clim.index.name = "month"
    return clim


def get_monthly_means(dataframe: pd.DataFrame) -> pd.DataFrame:
    monthly_means = dataframe.groupby([dataframe.index.year, dataframe.index.month]).mean()
    monthly_means.index.names = ["year", "month"]
    return monthly_means


def get_hottest_years(dataframe: pd.DataFrame, year_count: int) -> pd.DataFrame:
    # Step 1 — Compute mean anomaly per year
    mean_anom_per_year = dataframe.groupby(level="year").mean()

    # Step 2 — Find the 5 hottest years
    return mean_anom_per_year.nlargest(year_count).index


def index_for_plotting(monthly_anomalies: pd.DataFrame) -> pd.DataFrame:
    # Step 3 — Convert MultiIndex (year, month) to a proper datetime index for plotting
    # Here we assume each data point is monthly, e.g. end of month
    time_index = [pd.Timestamp(year=int(y), month=int(m), day=15) for y, m in monthly_anomalies.index]
    monthly_anomalies.index = pd.DatetimeIndex(time_index)
    return monthly_anomalies


def plot_average_years(monthly_anomalies: pd.DataFrame) -> None:
    plt.plot(monthly_anomalies.index, monthly_anomalies.values, color="lightgray", linewidth=1.5, label="_nolegend_")


def plot_hottest_years(monthly_anomalies: pd.DataFrame, hottest_years) -> None:
    for year in hottest_years:
        mask = monthly_anomalies.index.year == year
        plt.plot(monthly_anomalies.index[mask], monthly_anomalies[mask], color="red", linewidth=2.5, label=str(year))


def generate_graph_one(dataframe: pd.DataFrame, export_name="Anomalies from 1991 to 2020.png"):
    RANGE_START, RANGE_END = "1991-01-01", "2020-12-31"
    df_date_restricted = dataframe[RANGE_START:RANGE_END]

    all_monthly_anomalies = get_monthly_means(df_date_restricted) - get_monthly_aggregates(df_date_restricted)
    monthly_anomalies = all_monthly_anomalies["tl_mittel"]
    hottest_years = get_hottest_years(monthly_anomalies, 5)

    monthly_anomalies = index_for_plotting(monthly_anomalies)

    plt.figure(figsize=(12, 6))
    plot_average_years(monthly_anomalies)
    plot_hottest_years(monthly_anomalies, hottest_years)
    plt.title("Monthly Temperature Anomalies (1991–2020 baseline)")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.legend(title="Top 5 Hottest Years")
    plt.grid(True, alpha=0.3)

    plt.savefig(export_name)


def render_deciles_quartiles_median(monthly_stats: pd.DataFrame, months, ax, var, title, color) -> None:
    # Interdecile (10–90%)
    ax.fill_between(
        months,
        monthly_stats[(var, 'p10')],
        monthly_stats[(var, 'p90')],
        color=color, alpha=0.15, label='10–90% range'
    )

    # Interquartile (25–75%)
    ax.fill_between(
        months,
        monthly_stats[(var, 'p25')],
        monthly_stats[(var, 'p75')],
        color=color, alpha=0.3, label='25–75% range'
    )

    # Median line
    ax.plot(
        months,
        monthly_stats[(var, 'median')],
        color=color, linewidth=2.5, label='Median'
    )

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

def generate_graph_two(dataframe: pd.DataFrame, birth_year: str, export_name="Hot days and Tropical nights.png"):
    RANGE_START = "2023-01-01"
    df_after_2023 = dataframe[RANGE_START:]
    df_birth_year = dataframe[f"{birth_year}-01-01":f"{birth_year}-12-31"]
    df_date_restricted = pd.concat([df_after_2023, df_birth_year])

    DESIRED_STATISTICS = ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75),
                      lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)]
    VARIABLES = ['tl_mittel', 'tlmin', 'tlmax']
    DESIRED_STATISTIC_NAMES = ['median', 'p25', 'p75', 'p10', 'p90']
    # Group by month number (1–12)
    statistics_for_each_variable = {var: DESIRED_STATISTICS for var in VARIABLES}
    monthly_stats = df_date_restricted.groupby(df_date_restricted.index.month).agg(statistics_for_each_variable)

    # Clean up the column names
    monthly_stats.columns = pd.MultiIndex.from_product([VARIABLES, DESIRED_STATISTIC_NAMES])
    monthly_stats.index.name = 'month'

    # Create 3 subplots (min, mean, max)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    TITLES = ['Mean Temperature', 'Minimum Temperature', 'Maximum Temperature']
    COLORS = ['green', 'blue', 'red']

    MONTHS = range(1, 13)
    for ax, var, title, color in zip(axes, VARIABLES, TITLES, COLORS):
        render_deciles_quartiles_median(monthly_stats, MONTHS, ax, var, title, color)

    # Shared x-axis
    axes[-1].set_xlabel("Month")
    axes[-1].set_xticks(MONTHS)

    MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[-1].set_xticklabels(MONTH_LABELS)

    plt.suptitle(f"Monthly Temperature Distributions ({birth_year} and 2023-2025 baseline)", fontsize=15)
    plt.tight_layout(rect=(0.0, 0.0, 0.99, 1))
    plt.savefig(export_name)


def generate_graph_three(dataframe: pd.DataFrame, export_name="Monthly Temperature Distributions.png"):
    # Phase 3: Hot and cold days
    VARIABLES = ['tlmax', 'tlmin']
    THRESHOLD_TEMPS = [30, 20]
    d1 = {
        "tlmax": [lambda x: (x >= 30).sum()],
        "tlmin": [lambda x: (x >= 20).sum()]
    }
    d2 = {var: [lambda x, t=temp: (x >= t).sum()] for var, temp in zip(VARIABLES, THRESHOLD_TEMPS)}

    for var, temp in zip(VARIABLES, THRESHOLD_TEMPS): print(var, temp)
    annual_stats = dataframe.groupby(dataframe.index.year).agg(d2)
    annual_stats.columns = pd.MultiIndex.from_product(
        [['hot_days', 'tropical_nights']]
    )

    plt.figure()
    plt.plot(annual_stats.index, annual_stats.hot_days, color="red", linewidth=1.5, label="Hot Days")
    plt.plot(annual_stats.index, annual_stats.tropical_nights, color="blue", linewidth=1.5, label="Tropical Nights")

    plt.title("Hot days and Tropical nights by year")
    plt.xlabel("Year")
    plt.ylabel("Hot Times")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(export_name)

CLIMATE_DATA_CSV_LOCATION = "Messstationen_Graz_Tagesdaten_v2_Datensatz_19220101_20251031.csv"
df = get_climate_data(CLIMATE_DATA_CSV_LOCATION)

#generate_graph_one(df, "Anomalies from 1991 to 2020.png")
#generate_graph_two(df, "2002", "Monthly Temperature Distributions.png")
generate_graph_three(df, "Monthly Temperature Distributions.png")
