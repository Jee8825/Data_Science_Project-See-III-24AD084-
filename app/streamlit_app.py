


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = '/Users/Jee/College/II Year/ Odd Sem(III)/Technical/Data Science/flight_delay_project/models/flight_delay_model.pkl'
SCALER_PATH = '/Users/Jee/College/II Year/ Odd Sem(III)/Technical/Data Science/flight_delay_project/models/scaler.pkl'
FLIGHTS_PATH = '/Users/Jee/College/II Year/ Odd Sem(III)/Technical/Data Science/flight_delay_project/data/new_flights.csv'

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load flight data
df = pd.read_csv(FLIGHTS_PATH)
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# -----------------------------
# Streamlit Title
# -----------------------------
st.title("✈️ Flight Delay Prediction App")
st.markdown("Predict flight delays and visualize airport congestion.")

# -----------------------------
# Single Flight Prediction
# -----------------------------
st.subheader("Single Flight Prediction")

dep_time = st.number_input("CRS_DEP_TIME (HHMM)", 0, 2359, 800)
distance = st.number_input("Distance (miles)", 1, 5000, 500)
taxi_out = st.number_input("Taxi Out Time (minutes)", 0, 100, 15)
taxi_in = st.number_input("Taxi In Time (minutes)", 0, 100, 10)

features = ['CRS_DEP_TIME','DISTANCE','TAXI_OUT','TAXI_IN']

if st.button("Predict Delay"):
    X_input = np.array([[dep_time, distance, taxi_out, taxi_in]])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]
    st.write(f"**Prediction:** {'Delayed' if pred==1 else 'On Time'}")
    st.write(f"**Probability of Delay:** {prob:.2f}")

# -----------------------------
# Batch Prediction
# -----------------------------
st.subheader("Batch Prediction for Multiple Flights")
uploaded_file = st.file_uploader("Upload CSV with flights", type="csv")

if uploaded_file is not None:
    df_batch = pd.read_csv(uploaded_file)
    X_batch = df_batch[features].values
    X_scaled_batch = scaler.transform(X_batch)
    df_batch['DELAY_PRED'] = model.predict(X_scaled_batch)
    df_batch['DELAY_PROB'] = model.predict_proba(X_scaled_batch)[:,1]
    
    st.dataframe(df_batch.head())
    
    csv = df_batch.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predicted_flights.csv',
        mime='text/csv'
    )
    st.success("✅ Predictions ready for download!")

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("Flight Delay Visualizations")

# Delay distribution
st.write("**Arrival Delay Distribution**")
st.bar_chart(df['ARR_DELAY'].clip(lower=0))

# Airport congestion map
st.write("**Airport Congestion Map**")
congestion = df.groupby('ORIGIN')['DEP_DELAY'].mean()
airport_coords = {
    'JFK': (40.6413, -73.7781),
    'LAX': (33.9416, -118.4085),
    'ORD': (41.9742, -87.9073),
    'ATL': (33.6407, -84.4277),
    'DFW': (32.8998, -97.0403)
}

m = folium.Map(location=[39.8283,-98.5795], zoom_start=4)
for airport, delay in congestion.items():
    lat, lon = airport_coords.get(airport,(0,0))
    color = 'green' if delay<10 else 'orange' if delay<30 else 'red'
    folium.CircleMarker(location=[lat, lon],
                        radius=10,
                        color=color,
                        fill=True,
                        fill_color=color,
                        popup=f"{airport}: Avg Delay {delay:.2f} mins").add_to(m)

st_folium(m, width=700, height=500)

# Daily average arrival delay
st.write("**Daily Average Arrival Delays**")
daily_delays = df.groupby('FL_DATE')['ARR_DELAY'].mean()
st.line_chart(daily_delays)