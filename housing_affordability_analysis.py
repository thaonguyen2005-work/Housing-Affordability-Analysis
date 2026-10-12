# %% [markdown]
# # Housing Affordability Dynamics: Price-to-Rent Ratios Across U.S. Metro Areas, 2020-2026
#
# Each `# %%` marks the start of a new Jupyter cell. If you're in VS Code or
# Jupyter with the "Python interactive" extension, this whole file will open
# as a notebook automatically. Otherwise, copy each cell (the code between
# one `# %%` and the next) into its own cell in your .ipynb file.

# %% [markdown]
# ## 1. Imports and setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import statsmodels.api as sm

# Plot styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Study window, as defined in the proposal
START_DATE = "2020-01-01"
END_DATE = "2026-08-31"
TOP_N_METROS = 100  # based on Zillow's SizeRank

# %% [markdown]
# ## 2. Load raw data
#
# Both files are Zillow Research's public CSVs. They download directly with
# pandas, no API key needed. Each row is one metro; each column after the
# ID columns is one month (wide format).

# %%
ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
ZORI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Metro_zori_uc_sfrcondomfr_sm_month.csv"
)

zhvi_raw = pd.read_csv(ZHVI_URL)
zori_raw = pd.read_csv(ZORI_URL)

print("ZHVI shape:", zhvi_raw.shape)
print("ZORI shape:", zori_raw.shape)
zhvi_raw.head()

# %% [markdown]
# ## 3. Reshape wide -> long
#
# We melt each dataset so there is one row per metro per month, then parse
# the date and filter down to the study window (Jan 2020 - Aug 2026).

# %%
ID_COLS = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]


def wide_to_long(df, value_name):
    """Reshape a Zillow wide-format CSV into long format: one row per
    metro per month."""
    id_cols_present = [c for c in ID_COLS if c in df.columns]
    date_cols = [c for c in df.columns if c not in id_cols_present]

    long_df = df.melt(
        id_vars=id_cols_present,
        value_vars=date_cols,
        var_name="date",
        value_name=value_name,
    )
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df = long_df.dropna(subset=["date"])
    long_df = long_df[
        (long_df["date"] >= START_DATE) & (long_df["date"] <= END_DATE)
    ]
    return long_df


zhvi_long = wide_to_long(zhvi_raw, "zhvi")
zori_long = wide_to_long(zori_raw, "zori")

print("ZHVI long shape:", zhvi_long.shape)
print("ZORI long shape:", zori_long.shape)

# %% [markdown]
# ## 4. Restrict to the top 100 metros by SizeRank
#
# SizeRank 0 is the national "United States" aggregate row in Zillow's
# files, not a metro, so we drop it before ranking.

# %%
top_metro_ids = (
    zhvi_raw[zhvi_raw["RegionName"] != "United States"]
    .sort_values("SizeRank")
    .head(TOP_N_METROS)["RegionID"]
)

zhvi_long = zhvi_long[zhvi_long["RegionID"].isin(top_metro_ids)]
zori_long = zori_long[zori_long["RegionID"].isin(top_metro_ids)]

print(f"Metros retained: {zhvi_long['RegionID'].nunique()} (ZHVI), "
      f"{zori_long['RegionID'].nunique()} (ZORI)")

# %% [markdown]
# ## 5. Merge ZHVI and ZORI into one monthly panel

# %%
panel = pd.merge(
    zhvi_long[["RegionID", "RegionName", "SizeRank", "date", "zhvi"]],
    zori_long[["RegionID", "date", "zori"]],
    on=["RegionID", "date"],
    how="inner",  # keep only metro-months present in both datasets
)

panel = panel.sort_values(["RegionName", "date"]).reset_index(drop=True)
print("Merged panel shape:", panel.shape)
panel.head()

# %% [markdown]
# ## 6. Handle missing data with a completeness rule
#
# ZORI has weaker coverage in smaller metros. We keep only metros with at
# least 90% non-missing monthly observations across the full study window,
# per the feasibility plan in the proposal.

# %%
expected_months = pd.date_range(START_DATE, END_DATE, freq="MS").shape[0]
completeness = (
    panel.groupby("RegionName")["zori"]
    .apply(lambda s: s.notna().sum() / expected_months)
)

complete_metros = completeness[completeness >= 0.90].index
dropped_metros = completeness[completeness < 0.90].index

print(f"Keeping {len(complete_metros)} metros; "
      f"dropping {len(dropped_metros)} for insufficient ZORI coverage:")
print(list(dropped_metros))

panel = panel[panel["RegionName"].isin(complete_metros)].copy()
panel = panel.dropna(subset=["zhvi", "zori"])  # drop any remaining gaps

# %% [markdown]
# ## 7. Feature engineering: annualized rent and price-to-rent ratio
#
# $AnnualRent_{it} = ZORI_{it} \times 12$
#
# $PTR_{it} = \dfrac{ZHVI_{it}}{AnnualRent_{it}}$

# %%
panel["annual_rent"] = panel["zori"] * 12
panel["price_to_rent"] = panel["zhvi"] / panel["annual_rent"]

panel[["RegionName", "date", "zhvi", "zori", "annual_rent", "price_to_rent"]].head()

# %% [markdown]
# ## 8. Cumulative growth and growth divergence per metro
#
# For each metro, compare the first and last available month in the study
# window to compute cumulative % growth in price and rent, then:
#
# $Divergence_i = PriceGrowth_i - RentGrowth_i$

# %%
def first_last_growth(group):
    group = group.sort_values("date")
    first, last = group.iloc[0], group.iloc[-1]
    price_growth = (last["zhvi"] / first["zhvi"] - 1) * 100
    rent_growth = (last["zori"] / first["zori"] - 1) * 100
    ptr_start = first["price_to_rent"]
    ptr_end = last["price_to_rent"]
    return pd.Series({
        "price_growth_pct": price_growth,
        "rent_growth_pct": rent_growth,
        "divergence": price_growth - rent_growth,
        "ptr_start": ptr_start,
        "ptr_end": ptr_end,
        "ptr_change": ptr_end - ptr_start,
    })


metro_summary = (
    panel.groupby("RegionName")
    .apply(first_last_growth)
    .reset_index()
    .sort_values("divergence", ascending=False)
)

metro_summary.head(10)

# %% [markdown]
# ## 9. Exploratory analysis: summary statistics

# %%
metro_summary[["price_growth_pct", "rent_growth_pct", "divergence",
               "ptr_start", "ptr_end"]].describe()

# %% [markdown]
# ## 10. Time-series analysis: national median price-to-rent ratio over time

# %%
monthly_median = (
    panel.groupby("date")[["zhvi", "zori", "price_to_rent"]]
    .median()
    .reset_index()
)

fig, ax = plt.subplots()
ax.plot(monthly_median["date"], monthly_median["price_to_rent"], color="steelblue")
ax.set_title("Median Price-to-Rent Ratio Across Top 100 Metros (2020-2026)")
ax.set_xlabel("Date")
ax.set_ylabel("Price-to-Rent Ratio")
plt.tight_layout()
plt.show()

# %% [markdown]
# Indexed comparison of home-value growth vs. rent growth (both set to 100
# at the start of the window), so the two series are directly comparable.

# %%
monthly_median["zhvi_index"] = (
    monthly_median["zhvi"] / monthly_median["zhvi"].iloc[0] * 100
)
monthly_median["zori_index"] = (
    monthly_median["zori"] / monthly_median["zori"].iloc[0] * 100
)

fig, ax = plt.subplots()
ax.plot(monthly_median["date"], monthly_median["zhvi_index"],
        label="Home value (indexed)", color="darkorange")
ax.plot(monthly_median["date"], monthly_median["zori_index"],
        label="Rent (indexed)", color="seagreen")
ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Indexed Home Value vs. Rent Growth (Jan 2020 = 100)")
ax.set_xlabel("Date")
ax.set_ylabel("Index (Jan 2020 = 100)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Cross-metro comparison

# %% [markdown]
# ### 11a. Scatterplot: price growth vs. rent growth
# Metros above the 45-degree line had price growth outpace rent growth.

# %%
fig, ax = plt.subplots()
sns.scatterplot(
    data=metro_summary, x="rent_growth_pct", y="price_growth_pct",
    hue="divergence", palette="coolwarm", ax=ax, legend=False
)
lims = [metro_summary[["rent_growth_pct", "price_growth_pct"]].min().min(),
        metro_summary[["rent_growth_pct", "price_growth_pct"]].max().max()]
ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1)
ax.set_title("Cumulative Price Growth vs. Rent Growth by Metro (2020-2026)")
ax.set_xlabel("Rent growth (%)")
ax.set_ylabel("Home value growth (%)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 11b. Ranked bar chart: top and bottom 15 metros by divergence

# %%
top15 = metro_summary.head(15)
bottom15 = metro_summary.tail(15)
ranked = pd.concat([top15, bottom15])

fig, ax = plt.subplots(figsize=(10, 10))
colors = ["firebrick" if v > 0 else "steelblue" for v in ranked["divergence"]]
ax.barh(ranked["RegionName"], ranked["divergence"], color=colors)
ax.set_title("Metros With Largest Price-Rent Growth Divergence")
ax.set_xlabel("Divergence = Price growth % - Rent growth %")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Metro pattern analysis: K-means clustering
#
# Group metros by their cumulative price growth and rent growth patterns.
# Features are standardized first since K-means is distance-based.

# %%
cluster_features = metro_summary[["price_growth_pct", "rent_growth_pct"]].copy()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# Elbow method to help choose k (inspect the plot, then set K below)
inertias = []
k_range = range(1, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_features)
    inertias.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(list(k_range), inertias, marker="o")
ax.set_title("Elbow Plot for Choosing K")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia")
plt.tight_layout()
plt.show()

# %%
K = 4  # update this based on where the elbow plot bends

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
metro_summary["cluster"] = kmeans.fit_predict(scaled_features)

fig, ax = plt.subplots()
sns.scatterplot(
    data=metro_summary, x="rent_growth_pct", y="price_growth_pct",
    hue="cluster", palette="tab10", s=80, ax=ax
)
ax.set_title(f"Metro Clusters by Price & Rent Growth Pattern (k={K})")
ax.set_xlabel("Rent growth (%)")
ax.set_ylabel("Home value growth (%)")
plt.tight_layout()
plt.show()

# Cluster profiles: helps you interpret and name each cluster in your report
metro_summary.groupby("cluster")[["price_growth_pct", "rent_growth_pct",
                                    "divergence"]].mean()

# %% [markdown]
# ## 13. Optional statistical extension
#
# Two versions, matching the proposal's "optional extension":

# %% [markdown]
# ### 13a. Cross-sectional regression
# Does rent growth statistically predict price growth across metros?

# %%
X = sm.add_constant(metro_summary["rent_growth_pct"])
y = metro_summary["price_growth_pct"]
model = sm.OLS(y, X).fit()
print(model.summary())

# %% [markdown]
# ### 13b. Lagged correlation (time-series version)
# Using the national monthly median series: does rent growth in month t
# correlate with home-value growth in month t+lag?

# %%
monthly_median["zhvi_mom"] = monthly_median["zhvi_index"].pct_change() * 100
monthly_median["zori_mom"] = monthly_median["zori_index"].pct_change() * 100

lag_results = {}
for lag in range(0, 7):  # test 0 to 6 month lags
    shifted_rent = monthly_median["zori_mom"].shift(lag)
    valid = monthly_median["zhvi_mom"].notna() & shifted_rent.notna()
    corr, pval = pearsonr(monthly_median.loc[valid, "zhvi_mom"],
                           shifted_rent[valid])
    lag_results[lag] = (corr, pval)

lag_df = pd.DataFrame(lag_results, index=["correlation", "p_value"]).T
print(lag_df)

# %% [markdown]
# ## 14. Export cleaned data and summary table for the report/README

# %%
panel.to_csv("metro_panel_2020_2026.csv", index=False)
metro_summary.to_csv("metro_summary_growth_divergence.csv", index=False)

print("Saved: metro_panel_2020_2026.csv, metro_summary_growth_divergence.csv")
