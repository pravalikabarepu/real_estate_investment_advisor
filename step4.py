"""STEP 5 - Streamlit app.   Run:  streamlit run app.py"""
import json
import re

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.common import CLEAN_CSV, MODEL_DIR, add_features

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV)


@st.cache_resource
def load_models():
    return joblib.load(MODEL_DIR / "classifier.joblib"), joblib.load(MODEL_DIR / "regressor.joblib")


def read_json(name):
    p = MODEL_DIR / name
    return json.load(open(p)) if p.exists() else None


df = load_data()
clf, reg = load_models()
clf_feats, reg_feats = list(clf.feature_names_in_), list(reg.feature_names_in_)

page = st.sidebar.radio("Go to", ["🔮 Predict", "🔎 Explore properties", "📊 Market insights", "🧠 Model details"])
st.sidebar.caption("Predictions are estimates based on a rule-based label and assumed city growth "
                   "rates. They are not financial advice.")


# ================================================================== PREDICT
if page == "🔮 Predict":
    st.title("🏠 Real Estate Investment Advisor")
    st.write("Enter the property details to see if it is a good investment and its estimated price in 5 years.")

    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", sorted(df["State"].unique()))
        city = st.selectbox("City", sorted(df.loc[df["State"] == state, "City"].unique()))
        ptype = st.selectbox("Property type", sorted(df["Property_Type"].unique()))
        bhk = st.number_input("BHK", 1, 10, int(df["BHK"].median()))
        size = st.number_input("Size (sq ft)", 100, 20000, int(df["Size_in_SqFt"].median()), step=50)
        price = st.number_input("Current price (Lakhs)", 1.0, 100000.0,
                                float(round(df["Price_in_Lakhs"].median(), 1)), step=5.0)
    with c2:
        furnished = st.selectbox("Furnished status", sorted(df["Furnished_Status"].unique()))
        total_floors = st.number_input("Total floors in building", 1, 100, int(df["Total_Floors"].median()))
        floor_no = st.number_input("Floor number", 0, 100, min(int(df["Floor_No"].median()), int(total_floors)))
        age = st.number_input("Age of property (years)", 0, 100, int(df["Age_of_Property"].median()))
        parking = st.number_input("Parking spaces", 0, 20, int(df["Parking_Space"].median()))
        facing = st.selectbox("Facing", sorted(df["Facing"].unique()))
    with c3:
        schools = st.number_input("Nearby schools", 0, 50, int(df["Nearby_Schools"].median()))
        hospitals = st.number_input("Nearby hospitals", 0, 50, int(df["Nearby_Hospitals"].median()))
        transport = st.selectbox("Public transport accessibility", sorted(df["Public_Transport_Accessibility"].unique()))
        security = st.selectbox("Security", sorted(df["Security"].unique()))
        owner = st.selectbox("Owner type", sorted(df["Owner_Type"].unique()))
        availability = st.selectbox("Availability status", sorted(df["Availability_Status"].unique()))

    all_amen = sorted({a.strip() for s in df["Amenities"].dropna().unique()
                       for a in re.split(r"[,;|]", str(s)) if a.strip() and a.strip() != "None"})
    amenities = st.multiselect("Amenities", all_amen)

    if st.button("Analyse this property", type="primary"):
        if floor_no > total_floors:
            st.error("Floor number cannot be higher than total floors.")
            st.stop()
        row = {"State": state, "City": city, "Property_Type": ptype, "BHK": bhk,
               "Size_in_SqFt": size, "Price_in_Lakhs": price, "Furnished_Status": furnished,
               "Floor_No": floor_no, "Total_Floors": total_floors, "Age_of_Property": age,
               "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
               "Public_Transport_Accessibility": transport, "Parking_Space": parking,
               "Security": security, "Amenities": ", ".join(amenities) or "None",
               "Facing": facing, "Owner_Type": owner, "Availability_Status": availability}
        X = add_features(pd.DataFrame([row]))

        proba = clf.predict_proba(X[clf_feats])[0]
        is_good = int(proba[1] >= 0.5)
        confidence = float(max(proba))
        future = float(reg.predict(X[reg_feats])[0])
        cagr = (future / price) ** (1 / 5) - 1

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Investment verdict", "✅ Good Investment" if is_good else "⚠️ Not a Good Investment")
        r2.metric("Model confidence", f"{confidence:.0%}")
        r3.metric("Estimated price after 5 years", f"₹ {future:,.1f} L", f"{future - price:+,.1f} L")
        st.progress(confidence)

        ppsf = price * 100000 / size
        city_med = df.loc[df["City"] == city, "Price_per_SqFt"].median()
        st.write(f"**Price per sq ft:** ₹{ppsf:,.0f} vs city median ₹{city_med:,.0f} "
                 f"({'below' if ppsf <= city_med else 'above'} the city median). "
                 f"Implied yearly growth: **{cagr:.1%}**.")

        years = list(range(0, 6))
        trend = pd.DataFrame({"Year": years, "Estimated value (Lakhs)": [price * (1 + cagr) ** t for t in years]})
        st.plotly_chart(px.line(trend, x="Year", y="Estimated value (Lakhs)", markers=True,
                                title="Projected value over 5 years"))


# ================================================================== EXPLORE
elif page == "🔎 Explore properties":
    st.title("🔎 Explore properties")
    f1, f2, f3, f4 = st.columns(4)
    states = f1.multiselect("State", sorted(df["State"].unique()))
    cities = f2.multiselect("City", sorted(df.loc[df["State"].isin(states), "City"].unique()
                                           if states else df["City"].unique()))
    bhks = f3.multiselect("BHK", sorted(df["BHK"].unique()))
    only_good = f4.checkbox("Only good investments")
    lo, hi = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
    price_rng = st.slider("Price range (Lakhs)", lo, hi, (lo, hi))
    slo, shi = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
    size_rng = st.slider("Size range (sq ft)", slo, shi, (slo, shi))

    view = df[df["Price_in_Lakhs"].between(*price_rng) & df["Size_in_SqFt"].between(*size_rng)]
    if states: view = view[view["State"].isin(states)]
    if cities: view = view[view["City"].isin(cities)]
    if bhks: view = view[view["BHK"].isin(bhks)]
    if only_good: view = view[view["Good_Investment"] == 1]

    st.write(f"**{len(view):,} properties match**")
    show = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt", "Availability_Status", "Good_Investment", "Future_Price_5Y"]
    st.dataframe(view[show].head(1000))
    st.download_button("Download filtered results (CSV)", view[show].to_csv(index=False), "filtered_properties.csv")


# ================================================================== INSIGHTS
elif page == "📊 Market insights":
    st.title("📊 Market insights")
    t1, t2, t3, t4 = st.tabs(["Prices by city", "Location heatmap", "Trends", "Investment & correlations"])

    with t1:
        g = df.groupby("City")[["Price_in_Lakhs", "Price_per_SqFt"]].mean().reset_index()
        metric = st.radio("Metric", ["Price_in_Lakhs", "Price_per_SqFt"], horizontal=True)
        st.plotly_chart(px.bar(g.sort_values(metric, ascending=False), x="City", y=metric,
                               title=f"Average {metric} by city"))

    with t2:
        pivot = df.pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", color_continuous_scale="YlGnBu",
                                  title="Average price per sq ft: city × property type"))

    with t3:
        city = st.selectbox("City", sorted(df["City"].unique()))
        tr = df[df["City"] == city].groupby("Year_Built")["Price_per_SqFt"].mean().reset_index()
        st.plotly_chart(px.line(tr, x="Year_Built", y="Price_per_SqFt",
                                title=f"{city}: price per sq ft by year built"))

    with t4:
        gi = df.groupby("City")["Good_Investment"].mean().reset_index().sort_values("Good_Investment", ascending=False)
        st.plotly_chart(px.bar(gi, x="City", y="Good_Investment", title="Share of good investments by city"))
        num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
               "Nearby_Schools", "Nearby_Hospitals", "Parking_Space", "Infrastructure_Score", "Future_Price_5Y"]
        st.plotly_chart(px.imshow(df[num].corr(), text_auto=".2f", color_continuous_scale="RdYlGn",
                                  zmin=-1, zmax=1, title="Correlation matrix"))


# ================================================================== MODEL INFO
else:
    st.title("🧠 Model details")
    for title, mfile, ifile in [("Classification: Good Investment", "clf_metrics.json", "clf_importance.csv"),
                                ("Regression: Price after 5 years", "reg_metrics.json", "reg_importance.csv")]:
        st.subheader(title)
        m = read_json(mfile)
        if m:
            st.caption(f"Best model: **{m['best']}**")
            st.dataframe(pd.DataFrame(m["models"]).T)
        p = MODEL_DIR / ifile
        if p.exists():
            imp = pd.read_csv(p).head(15).iloc[::-1]
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                                   title="Top 15 features"))