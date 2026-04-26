
# AI Assisted Triage Support Tool
# Improved Streamlit prototype interface


import streamlit as st
import pandas as pd
import joblib


# Load trained model and saved feature columns


model = joblib.load("CMP6202_Triage_Final_Model.pkl")
model_columns = joblib.load("model_columns.pkl")


# Page settings


st.set_page_config(
    page_title="AI Assisted Triage Support Tool",
    page_icon="🩺",
    layout="wide"
)


# Custom styling to make the interface look cleaner


st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .stApp {
            background-color: #f7f9fc;
        }
        h1, h2, h3 {
            color: #1f2d3d;
        }
        .info-box {
            background-color: #e8f1fb;
            padding: 15px;
            border-radius: 10px;
            border-left: 6px solid #2c7be5;
            margin-bottom: 20px;
        }
        .high-box {
            background-color: #fdeaea;
            padding: 18px;
            border-radius: 10px;
            border-left: 6px solid #d9534f;
            margin-top: 15px;
            margin-bottom: 15px;
        }
        .medium-box {
            background-color: #fff4dd;
            padding: 18px;
            border-radius: 10px;
            border-left: 6px solid #f0ad4e;
            margin-top: 15px;
            margin-bottom: 15px;
        }
        .low-box {
            background-color: #eaf7ea;
            padding: 18px;
            border-radius: 10px;
            border-left: 6px solid #5cb85c;
            margin-top: 15px;
            margin-bottom: 15px;
        }
        .small-note {
            font-size: 14px;
            color: #5f6b7a;
        }
    </style>
""", unsafe_allow_html=True)


# Header


st.title("🩺 AI Assisted Triage Support Tool")

st.markdown("""
<div class="info-box">
This prototype predicts a <b>KTAS triage level</b> and converts it into a simpler
<b>urgency category</b> based on selected patient triage information.
</div>
""", unsafe_allow_html=True)

st.caption("This prototype is for decision support only and does not replace clinical judgement.")


# Layout in two columns


col1, col2 = st.columns([1.2, 1])


# Left column: input form


with col1:
    st.subheader("Patient Input")

    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    patients_per_hour = st.number_input("Patients number per hour", min_value=0, max_value=30, value=5)
    arrival_mode = st.number_input("Arrival mode", min_value=1, max_value=7, value=2)
    injury = st.selectbox("Injury", [0, 1], help="0 = No, 1 = Yes")
    mental = st.number_input("Mental", min_value=1, max_value=4, value=1)
    pain = st.selectbox("Pain", [0, 1], help="0 = No, 1 = Yes")
    saturation = st.number_input("Saturation", min_value=50.0, max_value=100.0, value=98.0)
    age_bracket = st.selectbox("Age bracket", [0, 1, 2, 3], help="0=0–18, 1=19–40, 2=41–60, 3=61+")


# Right column: explanation panel


with col2:
    st.subheader("Model Insight")

    st.markdown("""
**Most important features in the final model included:**
- Age  
- Pain score related variables  
- Patients number per hour  
- Injury status  
- Arrival mode  
- Oxygen saturation  
- Mental state  

These variables reflect clinically meaningful patterns that are relevant to emergency triage decision making.
""")

    st.markdown("""
<div class="small-note">
The model was retrained after removing leakage features so that predictions better reflect a realistic triage scenario.
</div>
""", unsafe_allow_html=True)


# Build an empty input row using saved model columns


input_data = pd.DataFrame(0, index=[0], columns=model_columns)


# Fill selected fields only


if "Age" in input_data.columns:
    input_data.at[0, "Age"] = age

if "Patients number per hour" in input_data.columns:
    input_data.at[0, "Patients number per hour"] = patients_per_hour

if "Arrival mode" in input_data.columns:
    input_data.at[0, "Arrival mode"] = arrival_mode

if "Injury" in input_data.columns:
    input_data.at[0, "Injury"] = injury

if "Mental" in input_data.columns:
    input_data.at[0, "Mental"] = mental

if "Pain" in input_data.columns:
    input_data.at[0, "Pain"] = pain

if "Saturation" in input_data.columns:
    input_data.at[0, "Saturation"] = saturation

if "age_bracket" in input_data.columns:
    input_data.at[0, "age_bracket"] = age_bracket


# Prediction section


st.markdown("---")

if st.button("Predict Triage Level"):

    prediction = model.predict(input_data)[0]

    if prediction in [1, 2]:
        urgency = "High Urgency"
        st.markdown(f"""
        <div class="high-box">
            <h3>Predicted KTAS Level: {prediction}</h3>
            <h3>Urgency Category: {urgency}</h3>
            <p>This result suggests that the patient may require rapid clinical attention.</p>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == 3:
        urgency = "Medium Urgency"
        st.markdown(f"""
        <div class="medium-box">
            <h3>Predicted KTAS Level: {prediction}</h3>
            <h3>Urgency Category: {urgency}</h3>
            <p>This result suggests moderate urgency and continued clinical assessment.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        urgency = "Low Urgency"
        st.markdown(f"""
        <div class="low-box">
            <h3>Predicted KTAS Level: {prediction}</h3>
            <h3>Urgency Category: {urgency}</h3>
            <p>This result suggests lower immediate urgency, although clinical judgement remains essential.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Input Summary")
    st.dataframe(input_data.loc[:, (input_data != 0).any(axis=0)])


# Footer


st.markdown("---")
st.caption("Developed as part of CMP6202 Individual Honours Project.")