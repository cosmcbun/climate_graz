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

df = pd.read_csv("Messstationen_Graz_Tagesdaten_v2_Datensatz_19220101_20251031.csv")
df["time"] = pd.to_datetime(df.time)
df = df.set_index("time")

# Phase one: Climatologies
df_date_restricted = df["1991-01-01":"2020-12-31"]
yearly_means = df_date_restricted.groupby(df_date_restricted.index.year).mean().dropna()

# Get mean value for each month (climatology)
df_clim = df_date_restricted["1991-01":"2020-12"]
clim = df_clim.groupby(df_clim.index.month).mean()
clim.index.name = "month"

# Then: monthly means, and subtract climatology of each month from the monthly means to get monthly anomalies.
monthly_means = df_date_restricted.groupby([df_date_restricted.index.year, df_date_restricted.index.month]).mean()
monthly_means.index.names = ["year", "month"]

anom = monthly_means - clim

series = anom["tl_mittel"]

# Step 1 — Compute mean anomaly per year
mean_anom_per_year = series.groupby(level="year").mean()

# Step 2 — Find the 5 hottest years
hottest_years = mean_anom_per_year.nlargest(5).index

# Step 3 — Convert MultiIndex (year, month) to a proper datetime index for plotting
# Here we assume each data point is monthly, e.g. end of month
time_index = [pd.Timestamp(year=int(y), month=int(m), day=15) for y, m in series.index]
series.index = pd.DatetimeIndex(time_index)

# Step 4 — Plot timeline and highlight hottest years
plt.figure(figsize=(12,6))
plt.plot(series.index, series.values, color="lightgray", linewidth=1.5, label="_nolegend_")

for year in hottest_years:
    mask = series.index.year == year
    plt.plot(series.index[mask], series[mask], color="red", linewidth=2.5, label=str(year))

plt.title("Monthly Temperature Anomalies (1991–2020 baseline)")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.legend(title="Top 5 Hottest Years")
plt.grid(True, alpha=0.3)
plt.savefig("Anomalies from 1991 to 2020.png")







# Phase two: medians

df_custom_years = pd.concat([df["2002-01-01":"2002-12-31"], df["2023-01-01":]])


# Group by month number (1–12)
monthly_stats = df_custom_years.groupby(df_custom_years.index.month).agg({
    "tl_mittel": ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75),
                  lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)],
    "tlmin":    ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75),
                  lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)],
    "tlmax":    ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75),
                  lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)]
})

# Clean up the column names
monthly_stats.columns = pd.MultiIndex.from_product(
    [['tl_mittel', 'tlmin', 'tlmax'],
     ['median', 'p25', 'p75', 'p10', 'p90']]
)
monthly_stats.index.name = 'month'

# Time to start displaying
months = range(1, 13)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Create 3 subplots (min, mean, max)
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
variables = ['tlmin', 'tl_mittel', 'tlmax']
titles = ['Minimum Temperature', 'Mean Temperature', 'Maximum Temperature']
colors = ['blue', 'green', 'red']  # blue, orange, red

for ax, var, title, color in zip(axes, variables, titles, colors):
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

# Shared x-axis
axes[-1].set_xlabel("Month")
axes[-1].set_xticks(months)
axes[-1].set_xticklabels(month_labels)

plt.suptitle("Monthly Temperature Distributions (1991–2020 baseline)", fontsize=15)
plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
plt.savefig("Monthly Temperature Distributions.png")




# Phase 3: Hot and cold days

annual_stats = df.groupby(df.index.year).agg({
    "tlmax": [lambda x: (x >= 30).sum()],
    "tlmin": [lambda x: (x >= 20).sum()]
})
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
plt.savefig("Hot days and Tropical nights.png")
