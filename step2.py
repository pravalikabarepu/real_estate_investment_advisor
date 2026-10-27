"""STEP 2 - Answers all 20 EDA questions and saves one chart per question.
Run:  python -m src.eda      (after src.preprocess)
Charts are saved in eda_outputs/. Read the printed numbers to write your insights.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common import EDA_DIR, RAW_CSV, load_clean

sns.set_theme(style="whitegrid")
EDA_DIR.mkdir(exist_ok=True)
df = load_clean()
sample = df.sample(min(len(df), 5000), random_state=42)


def save(name):
    plt.tight_layout()
    plt.savefig(EDA_DIR / f"{name}.png", dpi=120)
    plt.close()


def bar_mean(col, target, name, title, top=None, sort=True, rot=45):
    """Bar chart of the average of `target` for every value of `col`."""
    g = df.groupby(col)[target].mean()
    g = g.sort_values(ascending=False) if sort else g.sort_index()
    if top:
        g = g.head(top)
    print(f"\n{title}\n", g.round(2).to_string())
    plt.figure(figsize=(9, 4.5))
    g.plot(kind="bar", color="#2e7d5b")
    plt.title(title); plt.ylabel(target); plt.xticks(rotation=rot, ha="right")
    save(name)


# ============================ Q1-5  Price & size ============================
plt.figure(figsize=(8, 4)); sns.histplot(df["Price_in_Lakhs"], bins=50, kde=True)
plt.title("Q1 Distribution of property prices (Lakhs)"); save("q01_price_distribution")
print("Q1 price summary:\n", df["Price_in_Lakhs"].describe().round(2))

plt.figure(figsize=(8, 4)); sns.histplot(df["Size_in_SqFt"], bins=50, kde=True)
plt.title("Q2 Distribution of property sizes (SqFt)"); save("q02_size_distribution")

plt.figure(figsize=(9, 4.5)); sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt")
plt.title("Q3 Price per SqFt by property type"); save("q03_ppsf_by_type")

plt.figure(figsize=(7, 5)); sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.4)
plt.title("Q4 Size vs price"); save("q04_size_vs_price")
print("\nQ4 correlation size vs price:", round(df["Size_in_SqFt"].corr(df["Price_in_Lakhs"]), 3))

# Q5 uses the RAW file, because the cleaned file already has outliers capped
raw = pd.read_csv(RAW_CSV); raw.columns = raw.columns.str.strip()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, c in zip(ax, ["Price_per_SqFt", "Size_in_SqFt"]):
    sns.boxplot(y=raw[c], ax=a); a.set_title(c)
    q1, q3 = raw[c].quantile([.25, .75]); iqr = q3 - q1
    n = ((raw[c] < q1 - 1.5 * iqr) | (raw[c] > q3 + 1.5 * iqr)).sum()
    print(f"Q5 outliers in {c} (1.5*IQR rule): {n}")
fig.suptitle("Q5 Outliers (raw data)"); save("q05_outliers")

# ============================ Q6-10  Location ===============================
bar_mean("State", "Price_per_SqFt", "q06_ppsf_by_state", "Q6 Average price per SqFt by state")
bar_mean("City", "Price_in_Lakhs", "q07_price_by_city", "Q7 Average price by city (top 20)", top=20)

g = df.groupby("Locality")["Age_of_Property"].median().sort_values(ascending=False).head(15)
print("\nQ8 median age by locality (top 15):\n", g.to_string())
plt.figure(figsize=(9, 4.5)); g.plot(kind="bar", color="#2e7d5b")
plt.title("Q8 Median property age by locality (15 oldest)"); plt.xticks(rotation=45, ha="right")
save("q08_age_by_locality")

ct = pd.crosstab(df["City"], df["BHK"])
ct.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="viridis")
plt.title("Q9 BHK distribution across cities"); plt.xticks(rotation=45, ha="right"); save("q09_bhk_by_city")

top5 = df.groupby("Locality")["Price_in_Lakhs"].median().nlargest(5).index
sub = df[df["Locality"].isin(top5)].copy()
sub["Built_Period"] = (sub["Year_Built"] // 5 * 5).astype(int)
trend = sub.groupby(["Built_Period", "Locality"])["Price_per_SqFt"].mean().unstack()
print("\nQ10 top 5 most expensive localities:", list(top5))
trend.plot(marker="o", figsize=(9, 4.5))
plt.title("Q10 Price per SqFt trend, top-5 expensive localities (by year built)"); save("q10_top5_locality_trend")

# ============================ Q11-15  Correlations ==========================
num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
       "Nearby_Schools", "Nearby_Hospitals", "Parking_Space", "Infrastructure_Score",
       "Amenity_Count", "Future_Price_5Y"]
plt.figure(figsize=(10, 7)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="RdYlGn", center=0)
plt.title("Q11 Correlation between numeric features"); save("q11_correlation")

bar_mean("Nearby_Schools", "Price_per_SqFt", "q12_schools_vs_ppsf", "Q12 Nearby schools vs price per SqFt", sort=False, rot=0)
bar_mean("Nearby_Hospitals", "Price_per_SqFt", "q13_hospitals_vs_ppsf", "Q13 Nearby hospitals vs price per SqFt", sort=False, rot=0)
bar_mean("Furnished_Status", "Price_in_Lakhs", "q14_price_by_furnishing", "Q14 Average price by furnished status", rot=0)
bar_mean("Facing", "Price_per_SqFt", "q15_ppsf_by_facing", "Q15 Price per SqFt by facing direction", rot=0)

# ============================ Q16-20  Investment / amenities ================
for col, name, title in [("Owner_Type", "q16_owner_type", "Q16 Properties per owner type"),
                         ("Availability_Status", "q17_availability", "Q17 Properties per availability status")]:
    print(f"\n{title}\n", df[col].value_counts().to_string())
    plt.figure(figsize=(6, 4)); sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(title); save(name)

bar_mean("Parking_Space", "Price_in_Lakhs", "q18_parking_vs_price", "Q18 Parking spaces vs average price", sort=False, rot=0)
bar_mean("Amenity_Count", "Price_per_SqFt", "q19_amenities_vs_ppsf", "Q19 Number of amenities vs price per SqFt", sort=False, rot=0)
bar_mean("Public_Transport_Accessibility", "Price_per_SqFt", "q20_transport_vs_ppsf", "Q20 Public transport vs price per SqFt", rot=0)
bar_mean("Public_Transport_Accessibility", "Good_Investment", "q20b_transport_vs_investment", "Q20b Share of good investments by transport access", rot=0)

# ============================ Bonus charts from the project brief ===========
bar_mean("City", "Good_Investment", "bonus_good_investment_by_city", "Bonus: share of good investments by city", top=20)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=sample, x="Infrastructure_Score", y="Future_Price_5Y", alpha=0.4)
plt.title("Bonus: infrastructure score vs 5-year price"); save("bonus_infra_vs_future_price")

print("\nAll charts saved in", EDA_DIR)
