#fama french 3 factor model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm

# --- Visualization setup ---
plt.style.use("default")
params = {
    "axes.labelsize": 8, "font.size": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "grey", "axes.grid": True,  "grid.alpha": 0.5, "grid.linestyle": ":",
}
plt.rcParams.update(params)

# --- Step 1: Download QQQ price data from Yahoo Finance ---
qqq_daily = yf.download("QQQ", start="2006-01-01", end="2023-12-31", auto_adjust=True)

# Resample to month-end and calculate returns
qqq_monthly = qqq_daily[["Close"]].resample("ME").ffill()
qqq_monthly.index = qqq_monthly.index.to_period("M")
qqq_monthly["Return"] = qqq_monthly["Close"].pct_change() * 100
qqq_monthly.dropna(inplace=True)

# --- Step 2: Load Fama-French 3-Factor data ---
raw_df = pd.read_csv("../data/F-F_Research_Data_Factors.csv", skiprows=3)

# Rename the first column to 'Date'
raw_df.rename(columns={raw_df.columns[0]: "Date"}, inplace=True)

# Keep only rows where 'Date' matches YYYYMM format
raw_df = raw_df[raw_df["Date"].astype(str).str.match(r"^\d{6}$", na=False)]

# Convert 'Date' to datetime and set as index
raw_df["Date"] = pd.to_datetime(raw_df["Date"], format="%Y%m")
raw_df.set_index("Date", inplace=True)
raw_df.index = raw_df.index.to_period("M")

# Convert factor values to numeric
ff_factors_monthly = raw_df.apply(pd.to_numeric, errors="coerce").dropna()

# --- Step 3: Calculate excess returns of QQQ ---
# Align dates between QQQ and FF factors
common_index = qqq_monthly.index.intersection(ff_factors_monthly.index)
qqq_filtered = qqq_monthly.loc[common_index]
ff_factors_subset = ff_factors_monthly.loc[common_index]

# Excess return = QQQ return - RF
ff_factors_subset["Excess_Return"] = qqq_filtered["Return"] - ff_factors_subset["RF"]

# --- Step 4: Run the Fama-French regression ---
X = sm.add_constant(ff_factors_subset[["Mkt-RF", "SMB", "HML"]])
y = ff_factors_subset["Excess_Return"]
model = sm.OLS(y, X).fit()
print(model.summary())

# --- Step 5: Plotting coefficients and confidence intervals ---
factors = model.params.index[1:]
coefficients = model.params.values[1:]
ci = model.conf_int().iloc[1:]

ols_data = pd.DataFrame({
    "Factor": factors,
    "Coefficient": coefficients,
    "Confidence_Lower": ci[0].values,
    "Confidence_Upper": ci[1].values
})

plt.figure(figsize=(5, 4))
sns.barplot(x="Factor", y="Coefficient", data=ols_data, capsize=0.2, palette="coolwarm")
for i, row in ols_data.iterrows():
    plt.text(
        i, row["Coefficient"], f"p={model.pvalues[row['Factor']]:.4f}",
        ha="center", va="bottom", fontsize=7
    )
plt.title("Fama-French Factors on QQQ Monthly Returns (2006–2023)")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()
