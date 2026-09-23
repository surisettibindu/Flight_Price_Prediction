import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PAGE CONFIGURATION

st.set_page_config(page_title="Flight Price Prediction",page_icon="✈️",layout="wide")

# CUSTOM CSS

st.markdown("""<style>
    .main-title {font-size: 42px;font-weight: bold;text-align: center;margin-bottom: 10px;}
    .subtitle {text-align: center;color: #666;margin-bottom: 30px;}
    .metric-card {padding: 15px;border-radius: 10px;background-color: #f5f5f5;text-align: center;}</style>""", unsafe_allow_html=True)

# LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv("flightprice.csv")
try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please place ""`flightprice.csv` in the same folder as app.py.")
    st.stop()

# TITLE

st.markdown('<div class="main-title">✈️ Flight Price Prediction Dashboard</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">''Explore flight prices, analyze travel patterns, and predict ticket prices.''</div>',unsafe_allow_html=True)

# SIDEBAR

st.sidebar.header("🔎 Filters")
airlines = st.sidebar.multiselect("Airline",options=sorted(df["Airline"].unique()),default=sorted(df["Airline"].unique()))
classes = st.sidebar.multiselect("Travel Class",options=sorted(df["Class"].unique()),default=sorted(df["Class"].unique()))
seasons = st.sidebar.multiselect("Season",options=sorted(df["Season"].unique()),default=sorted(df["Season"].unique()))
stops = st.sidebar.multiselect("Number of Stops",options=sorted(df["Stops"].unique()),default=sorted(df["Stops"].unique()))

# FILTER DATA

filtered_df = df[(df["Airline"].isin(airlines)) &(df["Class"].isin(classes)) &(df["Season"].isin(seasons)) &(df["Stops"].isin(stops))]

# TOP METRICS

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("✈️ Flights",len(filtered_df))
with col2:
    st.metric("💰 Average Price",f"₹{filtered_df['Price_INR'].mean():,.0f}")
with col3:
    st.metric("📈 Maximum Price",f"₹{filtered_df['Price_INR'].max():,.0f}")
with col4:
    st.metric("📉 Minimum Price",f"₹{filtered_df['Price_INR'].min():,.0f}")
st.divider()

# TABS

tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset","📊 Analysis","📈 Visualizations","🤖 Price Prediction"])

# TAB 1 - DATASET

with tab1:

    st.subheader("📋 Flight Dataset")
    st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** flight records.")
    st.dataframe(filtered_df,use_container_width=True,hide_index=True)
    st.subheader("📌 Dataset Information")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("**Rows:**", df.shape[0])
        st.write("**Columns:**", df.shape[1])
        st.write("**Missing Values:**", df.isnull().sum().sum())

    with info_col2:
        st.write("**Average Price:**",f"₹{df['Price_INR'].mean():,.2f}")
        st.write("**Average Duration:**",f"{df['Duration_Minutes'].mean():.0f} minutes")
        st.write("**Average Distance:**",f"{df['Distance_KM'].mean():.0f} km")
    st.subheader("📥 Download Filtered Dataset")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV",data=csv,file_name="filtered_flight_dataset.csv",mime="text/csv")

# TAB 2 - ANALYSIS

with tab2:

    st.subheader("📊 Statistical Analysis")
    st.dataframe(filtered_df.describe(),use_container_width=True)
    st.subheader("💰 Average Price by Airline")
    airline_price = (filtered_df.groupby("Airline")["Price_INR"].mean().sort_values(ascending=False))
    st.dataframe(airline_price.reset_index(),use_container_width=True,hide_index=True)
    st.subheader("💺 Average Price by Class")
    class_price = (filtered_df.groupby("Class")["Price_INR"].mean().sort_values(ascending=False))
    st.dataframe(class_price.reset_index(),use_container_width=True,hide_index=True)
    st.subheader("🛑 Average Price by Number of Stops")
    stop_price = (filtered_df.groupby("Stops")["Price_INR"].mean())
    st.dataframe(stop_price.reset_index(),use_container_width=True,hide_index=True)

# TAB 3 - VISUALIZATIONS

with tab3:

    st.subheader("📈 Flight Price Visualizations")
    chart_type = st.selectbox("Select Visualization",["Average Price by Airline","Average Price by Class","Average Price by Stops","Price vs Distance","Price vs Booking Advance","Price Distribution"])

    # Airline Chart

    if chart_type == "Average Price by Airline":
        data = (filtered_df.groupby("Airline")["Price_INR"].mean().sort_values())
        st.bar_chart(data)

    # Class Chart

    elif chart_type == "Average Price by Class":
        data = (filtered_df.groupby("Class")["Price_INR"].mean().sort_values())
        st.bar_chart(data)

    # Stops Chart

    elif chart_type == "Average Price by Stops":
        data = (filtered_df.groupby("Stops")["Price_INR"].mean())
        st.bar_chart(data)

    # Distance vs Price

    elif chart_type == "Price vs Distance":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(filtered_df["Distance_KM"],filtered_df["Price_INR"])
        ax.set_title("Flight Price vs Distance")
        ax.set_xlabel("Distance (KM)")
        ax.set_ylabel("Price (INR)")
        st.pyplot(fig)

    # Booking Advance vs Price

    elif chart_type == "Price vs Booking Advance":
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(filtered_df["Booking_Advance_Days"],filtered_df["Price_INR"])
        ax.set_title("Flight Price vs Booking Advance")
        ax.set_xlabel("Booking Advance (Days)")
        ax.set_ylabel("Price (INR)")
        st.pyplot(fig)

    # Price Distribution

    elif chart_type == "Price Distribution":
        fig,ax = plt.subplots(figsize=(10, 5))
        ax.hist(filtered_df["Price_INR"],bins=15)
        ax.set_title("Flight Price Distribution")
        ax.set_xlabel("Price (INR)")
        ax.set_ylabel("Number of Flights")
        st.pyplot(fig)

# TAB 4 - MACHINE LEARNING

with tab4:

    st.subheader("🤖 Flight Price Prediction")
    st.write("Use the trained Random Forest model to estimate a flight ticket price.")

    # Features

    features = ["Airline","Source","Destination","Duration_Minutes","Stops","Class","Distance_KM","Booking_Advance_Days","Season","Baggage_KG"]
    target = "Price_INR"

    X = df[features]
    y = df[target]

    categorical_features = ["Airline","Source","Destination","Class","Season"]
    numerical_features = ["Duration_Minutes","Stops","Distance_KM","Booking_Advance_Days","Baggage_KG"]

    # Train Model
    
    preprocessor = ColumnTransformer(transformers=[("categorical",OneHotEncoder(handle_unknown="ignore"),categorical_features),("numerical","passthrough",numerical_features)])
    model = Pipeline(steps=[("preprocessor", preprocessor),("regressor",RandomForestRegressor(n_estimators=200,random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Model Metrics

    mae = mean_absolute_error(y_test,predictions)
    rmse = np.sqrt(mean_squared_error(y_test,predictions))
    r2 = r2_score(y_test,predictions)

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("MAE",f"₹{mae:,.0f}")

    with metric2:
        st.metric("RMSE",f"₹{rmse:,.0f}")

    with metric3:
        st.metric("R² Score",f"{r2:.3f}")

    st.divider()

    # Prediction Inputs

    st.subheader("🎯 Enter Flight Details")
    col1, col2 = st.columns(2)

    with col1:
        selected_airline = st.selectbox("Airline",sorted(df["Airline"].unique()))
        selected_source = st.selectbox("Source",sorted(df["Source"].unique()))
        selected_destination = st.selectbox("Destination",sorted(df["Destination"].unique()))
        selected_class = st.selectbox("Class",sorted(df["Class"].unique()))
        selected_season = st.selectbox("Season",sorted(df["Season"].unique()))

    with col2:
        duration = st.number_input("Duration (Minutes)",min_value=30,max_value=1000,value=150)
        selected_stops = st.number_input("Number of Stops",min_value=0,max_value=5,value=0)
        distance = st.number_input("Distance (KM)",min_value=100,max_value=5000,value=1000)
        advance_days = st.number_input("Booking Advance (Days)",min_value=0,max_value=365,value=30)
        baggage = st.number_input("Baggage (KG)",min_value=0,max_value=100,value=20)

    input_data = pd.DataFrame({
        "Airline": [selected_airline],
        "Source": [selected_source],
        "Destination": [selected_destination],
        "Duration_Minutes": [duration],
        "Stops": [selected_stops],
        "Class": [selected_class],
        "Distance_KM": [distance],
        "Booking_Advance_Days": [advance_days],
        "Season": [selected_season],
        "Baggage_KG": [baggage]})


    if st.button("💰 Predict Flight Price",type="primary"):
       predicted_price = model.predict(input_data)[0]
       st.success(f"Estimated Flight Price: ₹{predicted_price:,.0f}")
       st.info("This prediction is based on the synthetic dataset and should be treated as an educational estimate.")

st.divider()

st.caption(
    "✈️ Flight Price Prediction Dashboard"
    "Built with Python, Pandas, Scikit-learn and Streamlit")
