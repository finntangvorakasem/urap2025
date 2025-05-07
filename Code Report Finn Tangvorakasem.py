import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import re

# URAP Research Report: Economic Analysis for Public Policy - Finn Tangvorakasem Spring 2025

# Section 2: Data Structure and Methodology
# -----------------------------------------
# Load TED contracts in chunks and combine with Ukraine data
chunk_size = 100000
chunks = []
for chunk in pd.read_csv("export_CAN_2023_2018.csv", chunksize=chunk_size, low_memory=False):
    chunks.append(chunk)
data = pd.concat(chunks, ignore_index=True)
print(f"[Section 2] Combined TED data shape: {data.shape}")

#Sanity Check for Successful Loading
data.head(50).to_csv("finvalue.csv", index=False)

# Select key features
features = ["ID_NOTICE_CAN", "TED_NOTICE_URL", "ISO_COUNTRY_CODE",
            "WIN_NAME", "VALUE_EURO", "CPV", "NUMBER_OFFERS",
            "DT_AWARD", "YEAR"]

features_2 = ["ID_NOTICE_CAN", "TED_NOTICE_URL", "ISO_COUNTRY_CODE",
            "WIN_NAME", "VALUE_EURO_FIN_1", "CPV", "NUMBER_OFFERS",
            "DT_AWARD", "YEAR"]
eu_contract_new = data[features]

# Load Ukraine data and rename columns for consistency
data2 = pd.read_stata("UA_2018_2021.dta")
ukraine_contract = (
    data2.rename(columns={
        "lot_id": "ID_NOTICE_CAN",
        "win_name": "WIN_NAME",
        "auct_value": "VALUE_EURO",
        "cpv": "CPV",
        "iso_country_code": "ISO_COUNTRY_CODE"
    })
)
print(f"[Section 2] Ukraine data shape: {ukraine_contract.shape}")

# Combine EU and Ukraine
df = pd.concat([eu_contract_new, ukraine_contract], ignore_index=True)
print(f"[Section 2] Final combined dataset shape: {df.shape}\n")

df.head()

# Section 3: Key Data Anomalies
anomalies = df[(df["VALUE_EURO"] > 1e10) | (df["NUMBER_OFFERS"] == 999)]
anomalies.head()
len(anomalies)

df = df[(df["VALUE_EURO"] < 1e10) & (df['VALUE_EURO'] != 999)]

# Section 4: Winner Segmentation and Equal Contract Splitting
# Method 1: Regex-based splitting
df_split = df.copy()
df_split["WIN_NAME"] = df_split["WIN_NAME"].fillna("")

def split_names(winners):
    return [w.strip() for w in re.split(r"\s*(?:;|\||/|---|,|\n|\t)+\s*", winners) if w.strip()]

df_split["WIN_NAME_LIST"] = df_split["WIN_NAME"].apply(split_names)
df_split["NUM_WINNERS"] = df_split["WIN_NAME_LIST"].apply(len).replace(0, 1)
# explode and split values
df_split = (
    df_split.explode("WIN_NAME_LIST")
            .rename(columns={"WIN_NAME_LIST": "WIN_NAME"})
)
df_split["VALUE_EURO_SPLIT"] = df_split["VALUE_EURO"] / df_split["NUM_WINNERS"]
df_split.head()

# Method 2: Group-based splitting
cols = ["ID_NOTICE_CAN", "TED_NOTICE_URL", "VALUE_EURO"]
copy2 = df.copy()
copy2["supplier_count"] = copy2.groupby(cols)["WIN_NAME"].transform("count").replace(0,1)
copy2["VALUE_EURO_SPLIT"] = copy2["VALUE_EURO"] / copy2["supplier_count"]
copy2.head()

highest = copy2.sort_values("VALUE_EURO_SPLIT", ascending=False)
highest.head(200).to_csv("Highest_Value.csv")

lowest = copy2.sort_values("VALUE_EURO_SPLIT", ascending=True)
lowest.head(200).to_csv("Lowest_Value.csv")

middle = copy2.iloc[54800:55001]
middle.to_csv('Middle_Value.csv')

# Section 5: Low-Value, Symbolic Contracts and Framework Agreements
# Filter and categorize by value thresholds
new_data = df[df["WIN_NAME"].notna() & (df["WIN_NAME"].str.strip() != "")].copy()
supplier_counts = new_data.groupby('ID_NOTICE_CAN')['WIN_NAME'].count().rename('supplier_count')
new_data = new_data.merge(supplier_counts, on='ID_NOTICE_CAN')
new_data['VALUE_EURO_SPLIT'] = new_data['VALUE_EURO'] / new_data['supplier_count']
new_data = new_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['VALUE_EURO_SPLIT'])

counts = {
    '<=2000': len(new_data[new_data['VALUE_EURO'] <= 2000]),
    '<=1000': len(new_data[new_data['VALUE_EURO'] <= 1000]),
    '<=100':  len(new_data[new_data['VALUE_EURO'] <= 100])
}
print(f"[Section 5] Contract counts by value thresholds: {counts}")

under_1 = new_data[new_data['VALUE_EURO'] <= 1]
one_ten = new_data[(new_data['VALUE_EURO'] > 1) & (new_data['VALUE_EURO'] <= 10)]
ten_hundred = new_data[(new_data['VALUE_EURO'] >=10)&(new_data['VALUE_EURO']<=100)]
hundred_thousand = new_data[(new_data['VALUE_EURO']>=100)&(new_data['VALUE_EURO']<=1000)]

#Check for each country, change from 'DE' to others to check others.
de_group = (
    hundred_thousand[hundred_thousand['ISO_COUNTRY_CODE']=='DE']
    .groupby('CPV')
    .agg(contract_count=('ID_NOTICE_CAN','count'), total_value=('VALUE_EURO','sum'), avg_value=('VALUE_EURO','mean'))
    .reset_index()
    .sort_values('total_value', ascending=False)
)
de_group.to_csv('de_under1000.csv', index=False)

print(f"[Section 5] DE under-1000 grouped by CPV saved.\n")

# Section 6: Vague CPVs: Distribution and Risks
vague = df[~df['CPV'].astype(str).str.zfill(8).str.endswith('000000')]
print(f"[Section 6] Vague CPV records: {len(vague)}")
vague.head()

#%%
# Prepare working copy
df_raw = df.copy()
df_nodup = df.drop_duplicates(subset=["ID_NOTICE_CAN", "TED_NOTICE_URL"])

def calculate_vague_percentage(df_input, label):
    df_input = df_input.copy()
    df_input["is_vague"] = df_input["CPV"].astype(str).str.zfill(8).str.endswith("000000")
    country_stats = df_input.groupby("ISO_COUNTRY_CODE")["is_vague"].mean().reset_index()
    country_stats.columns = ["Country", label]
    country_stats[label] *= 100 
    return country_stats

# Calculate vague % per country for raw and no-duplicate data
vague_raw = calculate_vague_percentage(df_raw, "Raw Vague %")
vague_nodup = calculate_vague_percentage(df_nodup, "No-Duplicate Vague %")

# Merge Raw vs No-Duplicate Data
vague_compare = pd.merge(vague_nodup, vague_raw, on="Country")

def plot_vague(vague_compare):
    plt.figure(figsize=(10, 8))
    plt.scatter(vague_compare["No-Duplicate Vague %"], vague_compare["Raw Vague %"])

# Annotate points with country codes
    for _, row in vague_compare.iterrows():
        plt.text(row["No-Duplicate Vague %"], row["Raw Vague %"], row["Country"], fontsize=8)

# Add 45-degree reference line
    lims = [min(vague_compare["No-Duplicate Vague %"].min(), vague_compare["Raw Vague %"].min()) - 2,
        max(vague_compare["No-Duplicate Vague %"].max(), vague_compare["Raw Vague %"].max()) + 2]
    plt.plot(lims, lims, '--', color='gray')

    plt.title("Vague Contract Percentages by Country\n(Raw vs. No-Duplicate)")
    plt.xlabel("No-Duplicate Vague %")
    plt.ylabel("Raw Data Vague %")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#%%

def generate_vague_summary_table(df_input):
    df = df_input.copy()
    df["is_vague"] = df["CPV"].astype(str).str.zfill(8).str.endswith("000000")

    summary = (
        df.groupby("ISO_COUNTRY_CODE")
          .agg(
              TOTAL=("CPV", "count"),
              VAGUE=("is_vague", "sum")
          )
          .reset_index()
    )
    summary["VAGUE %"] = (summary["VAGUE"] / summary["TOTAL"]) * 100
    summary.columns = ["COUNTRY", "TOTAL", "VAGUE", "VAGUE %"]
    summary = summary.sort_values("VAGUE %", ascending=False).reset_index(drop=True)
    return summary




# Section 7: Competitiveness Analysis via Lorenz Curves
def compute_lorenz_curve(values):
    if len(values)<2:
        return np.array([]), np.array([])
    vals = np.sort(values)
    cum = np.cumsum(vals)/np.sum(vals)
    x = np.arange(1,len(vals)+1)/len(vals)
    return x, cum

# Plot for a sample country
def plot_lorenz_curve(country, df):
    auc_x, auc_y = compute_lorenz_curve(df[df['ISO_COUNTRY_CODE']==country]['VALUE_EURO'])
    win_x, win_y = compute_lorenz_curve(df[df['ISO_COUNTRY_CODE']==country].groupby('WIN_NAME')['VALUE_EURO'].sum().values)
    if len(auc_x)<2 or len(win_x)<2:
        print(f"Skipping {country}: insufficient data")
        return
    plt.figure(figsize=(8,6))
    plt.plot(auc_x*100, auc_y*100, label='Auctions')
    plt.plot(win_x*100, win_y*100, label='Winners')
    plt.plot([0,100],[0,100],'--',color='gray')
    plt.title(f"Lorenz Curve: {country}")
    plt.xlabel('Percent of Entities')
    plt.ylabel('Percent of Value')
    plt.legend(); plt.grid(True); plt.show()

plot_lorenz_curve('UK', new_data)
plot_lorenz_curve('UK', vague)

#Change in each input for different df results
def compute_granularity_score_single(df, digits=4, threshold=0.7, procurement_type='Usual'):
    
    df = df.copy()
    cpv_col = f'CPV_{digits}D'
    df[cpv_col] = df['CPV'].astype(str).str.zfill(8).str[:digits]

    results = []
    for country in df['ISO_COUNTRY_CODE'].dropna().unique():
        subset = df[(df['ISO_COUNTRY_CODE'] == country) & (df['PROCUREMENT_TYPE'] == procurement_type)]
        if len(subset) < 10:
            continue
        value_by_cpv = subset.groupby(cpv_col)['VALUE_EURO'].sum().sort_values(ascending=False)
        total = value_by_cpv.sum()
        if total == 0 or len(value_by_cpv) == 0:
            continue
        cumulative = value_by_cpv.cumsum()
        top = value_by_cpv[cumulative <= total * threshold]
        score = len(top) / subset[cpv_col].nunique()
        results.append({'Country': country, 'Score': score})

    return pd.DataFrame(results)

#Plot 1
def plot_granularity_score(score_df, digits=4, threshold=0.7, procurement_type='Usual'):
   
    plt.figure(figsize=(7, 6))
    plt.scatter(score_df['Score'], score_df['Score'])
    plt.title(f'Impact of CPV Granularity: {digits}-Digit ({procurement_type})')
    plt.xlabel(f'Competitiveness Score ({int(threshold*100)}/{int((1-threshold)*100)} – {digits} Digit)')
    plt.ylabel(f'Competitiveness Score ({int(threshold*100)}/{int((1-threshold)*100)} – {digits} Digit)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Plot 2 (in the report)
def plot_competitiveness_comparison(df_x, df_y, label_x, label_y, procurement_type='Usual'):
    
    merged = pd.merge(df_x, df_y, on='Country', suffixes=('_x', '_y'))

    plt.figure(figsize=(8, 6))
    plt.scatter(merged['Score_x'], merged['Score_y'])
    plt.title(f'Impact of CPV Granularity: {label_x} vs. {label_y} ({procurement_type})')
    plt.xlabel(f'Competitiveness Score ({label_x})')
    plt.ylabel(f'Competitiveness Score ({label_y})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#Example of Usage
score_usual_4_80 = compute_granularity_score_single(data, digits=4, threshold=0.8, procurement_type='Usual')
score_usual_5_80 = compute_granularity_score_single(data, digits=5, threshold=0.8, procurement_type='Usual')
score_usual_4_70 = compute_granularity_score_single(data, digits=4, threshold=0.7, procurement_type='Usual')
score_usual_5_70 = compute_granularity_score_single(data, digits=5, threshold=0.7, procurement_type='Usual')
#For Visuals
plot_competitiveness_comparison(
    score_usual_4_70,
    score_usual_5_70,
    label_x='70/30 - 4 Digit',
    label_y='70/30 - 5 Digit',
    procurement_type='Usual'
)

# Section 8: Auction-Winner Gap and Country Ranking
def compute_auction_winner_gap(df, country, top_pct=0.75):
    ax, ay = compute_lorenz_curve(df[df['ISO_COUNTRY_CODE']==country]['VALUE_EURO'])
    wx, wy = compute_lorenz_curve(df[df['ISO_COUNTRY_CODE']==country].groupby('WIN_NAME')['VALUE_EURO'].sum().values)
    if len(ax)<2 or len(wx)<2:
        return None
    common = np.linspace(0,1,100)
    ai = interp1d(ax, ay, fill_value='extrapolate')
    wi = interp1d(wx, wy, fill_value='extrapolate')
    gap = np.abs(ai(common)-wi(common))
    return trapezoid(gap, common)

gaps = {c: compute_auction_winner_gap(new_data,c) for c in new_data['ISO_COUNTRY_CODE'].unique()}
gap_df = pd.DataFrame.from_dict(gaps, orient='index', columns=['Gap']).dropna().reset_index()
gap_df.columns=['Country','Auction-Winner Gap']
ranked = gap_df.sort_values('Auction-Winner Gap').reset_index(drop=True)
print("[Section 8] Top 10 concentrated markets:")
print(ranked.head(10))

plt.figure(figsize=(12,6))
sns.barplot(data=ranked, x='Country', y='Auction-Winner Gap', color='blue')
plt.xticks(rotation=90); plt.title('Auction-Winner Gap by Country'); plt.tight_layout(); plt.show()

