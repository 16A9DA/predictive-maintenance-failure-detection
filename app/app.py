import streamlit as st
import joblib
import pandas as pd
import os

model = joblib.load(os.path.join("notebook", "final_model.joblib"))

st.title("Machine Failure Predictor")
st.markdown("Enter the machine's current sensor readings to predict failure risk.")

st.subheader("Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Product Quality Type",
        options=["L", "M", "H"],
        help="L = Low, M = Medium, H = High quality variant",
    )
    air_temp = st.slider(
        "Air Temperature (K)", min_value=295.0, max_value=305.0, value=300.0, step=0.1
    )
    process_temp = st.slider(
        "Process Temperature (K)",
        min_value=305.0,
        max_value=314.0,
        value=310.0,
        step=0.1,
    )

with col2:
    rotational_speed = st.slider(
        "Rotational Speed (rpm)", min_value=1168, max_value=2886, value=1538, step=1
    )
    torque = st.slider(
        "Torque (Nm)", min_value=3.8, max_value=76.6, value=40.0, step=0.1
    )
    tool_wear = st.slider(
        "Tool Wear (min)", min_value=0, max_value=253, value=108, step=1
    )

if st.button("Predict", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [
            {
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": machine_type,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(
            f"**Machine Failure Predicted** — {probability:.1%} failure probability"
        )
    else:
        st.success(f"**No Failure Predicted** — {probability:.1%} failure probability")

    st.progress(float(probability), text=f"Failure probability: {probability:.1%}")
