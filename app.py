import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# 1. Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Airline Passenger Satisfaction", page_icon="✈️")

st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write("Fill in the passenger and flight details, then click **Predict**.")


# ------------------------------------------------------------
# 2. Load the saved model (only once, not on every rerun)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    with open(BASE_DIR / "gradient_boosting_pipeline.pkl", "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'gradient_boosting_pipeline.pkl' not found. "
             "Keep it in the same folder as app.py.")
    st.stop()


# ------------------------------------------------------------
# 3. Input form
# ------------------------------------------------------------
st.subheader("Passenger and flight details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    age = st.number_input("Age", min_value=7, max_value=85, value=30)
    flight_distance = st.number_input("Flight Distance", min_value=31, max_value=5000, value=800)

with col2:
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=1600, value=0)
    arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, max_value=1600, value=0)

st.subheader("Service ratings")
st.caption("0 = not applicable, 1 = very poor, 5 = excellent")

# Same 14 rating columns, in the same order as the training data
rating_names = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
]

ratings = {}
left, right = st.columns(2)
for i, name in enumerate(rating_names):
    column = left if i % 2 == 0 else right
    with column:
        ratings[name] = st.slider(name, 0, 5, 3)


# ------------------------------------------------------------
# 4. Predict
# ------------------------------------------------------------
if st.button("Predict Satisfaction"):

    # Text -> numbers (same codes that LabelEncoder gave in Train.py)
    gender_code = {"Female": 0, "Male": 1}[gender]
    customer_code = {"Loyal Customer": 0, "disloyal Customer": 1}[customer_type]
    travel_code = {"Business travel": 0, "Personal Travel": 1}[travel_type]
    class_code = {"Business": 0, "Eco": 1, "Eco Plus": 2}[travel_class]

    # The extra feature we created in Train.py
    average_rating = sum(ratings.values()) / len(ratings)

    # One row, columns in the same order as X_train
    row = {
        "Gender": gender_code,
        "Customer Type": customer_code,
        "Age": age,
        "Type of Travel": travel_code,
        "Class": class_code,
        "Flight Distance": flight_distance,
    }
    row.update(ratings)
    row["Departure Delay in Minutes"] = departure_delay
    row["Arrival Delay in Minutes"] = arrival_delay
    row["Average Service Rating"] = average_rating

    data = pd.DataFrame([row])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    st.subheader("Result")
    if prediction == 1:
        st.success("Passenger is **Satisfied** 😊")
    else:
        st.error("Passenger is **Neutral or Dissatisfied** 😞")

    st.write(f"Probability of being satisfied: **{probability[1]:.1%}**")
    st.progress(float(probability[1]))
    st.write(f"Average service rating: **{average_rating:.2f}**")

    with st.expander("Show the data sent to the model"):
        st.dataframe(data)
