import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Airline Satisfaction", page_icon="✈️")

# Load model once and keep it in cache (not reloaded on every click)
@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).resolve().parent / "gradient_boosting_pipeline.pkl")

model = load_model()

# Same codes that LabelEncoder gave during training
GENDER = {"Female": 0, "Male": 1}
CUSTOMER = {"Loyal Customer": 0, "disloyal Customer": 1}
TRAVEL = {"Business travel": 0, "Personal Travel": 1}
CLASS = {"Business": 0, "Eco": 1, "Eco Plus": 2}

st.title("Airline Passenger Satisfaction App")
st.write("Enter passenger and flight details to predict satisfaction")

st.header("Passenger and Flight Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    age = st.number_input("Age", 7, 85, 30)
    flight_distance = st.number_input("Flight Distance", 31, 5000, 800)

with col2:
    type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    departure_delay = st.number_input("Departure Delay in Minutes", 0, 1600, 0)
    arrival_delay = st.number_input("Arrival Delay in Minutes", 0, 1600, 0)

st.header("Service Ratings")
st.caption("0 = not applicable, 1 = very poor, 5 = excellent")

left, right = st.columns(2)

with left:
    wifi = st.slider("Inflight wifi service", 0, 5, 3)
    booking = st.slider("Ease of Online booking", 0, 5, 3)
    food = st.slider("Food and drink", 0, 5, 3)
    seat = st.slider("Seat comfort", 0, 5, 3)
    onboard = st.slider("On-board service", 0, 5, 3)
    baggage = st.slider("Baggage handling", 0, 5, 3)
    inflight_service = st.slider("Inflight service", 0, 5, 3)

with right:
    time_conv = st.slider("Departure/Arrival time convenient", 0, 5, 3)
    gate = st.slider("Gate location", 0, 5, 3)
    boarding = st.slider("Online boarding", 0, 5, 3)
    entertainment = st.slider("Inflight entertainment", 0, 5, 3)
    legroom = st.slider("Leg room service", 0, 5, 3)
    checkin = st.slider("Checkin service", 0, 5, 3)
    clean = st.slider("Cleanliness", 0, 5, 3)

if st.button("Predict Satisfaction"):

    # Order must match the training columns exactly
    ratings = [wifi, time_conv, booking, gate, food, boarding, seat,
               entertainment, onboard, legroom, baggage, checkin,
               inflight_service, clean]

    average_rating = sum(ratings) / len(ratings)

    features = pd.DataFrame([[
        GENDER[gender], CUSTOMER[customer_type], age,
        TRAVEL[type_of_travel], CLASS[travel_class], flight_distance,
        *ratings, departure_delay, arrival_delay, average_rating,
    ]], columns=[
        "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance",
        "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink",
        "Online boarding", "Seat comfort", "Inflight entertainment",
        "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness",
        "Departure Delay in Minutes", "Arrival Delay in Minutes",
        "Average Service Rating",
    ])

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1] * 100)
    result = "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"

    st.subheader("Result")
    message = f"{result}\n\nSatisfaction Probability: {probability:.1f}%"
    if prediction == 1:
        st.success(message)
    else:
        st.error(message)
    st.progress(probability / 100)
