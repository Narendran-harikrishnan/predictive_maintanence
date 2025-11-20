import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# --- Constants ---
HF_MODEL_REPO_ID = "Narendranh/Narendran_PredictiveMaintenance-XGBoost-Model"
HF_MODEL_FILENAME = "xgboost_model.pkl"
INPUT_COLUMNS = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
    'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature'
]

# --- Function to Load Model from Hugging Face ---
# Use an aggressive layout (wide mode) and custom styling
st.set_page_config(
    page_title="Predictive App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, modern look
st.markdown("""
<style>
    /* Main Streamlit App container */
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 5rem;
        padding-left: 5%;
        padding-right: 5%;
    }
    /* Title styling */
    h1 {
        color: #FF4B4B; /* Streamlit's primary red */
        text-align: center;
        margin-bottom: 0.5rem;
    }
    h3 {
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Section dividers */
    .st-emotion-cache-1pxn4lb {
        border-top: 2px solid #ddd;
    }
    /* Custom Card for Results */
    .result-card {
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .result-card-success {
        background-color: #e6ffec; /* Light green */
        border-left: 8px solid #4CAF50;
    }
    .result-card-failure {
        background-color: #ffe6e6; /* Light red */
        border-left: 8px solid #F44336;
    }
    .result-card h2 {
        text-align: left;
        color: #333;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .st-emotion-cache-10xtr5v {
        background-color: #f0f2f6; /* Lighter background for inputs */
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Downloads the model artifact from the Hugging Face Hub and loads it."""
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=HF_MODEL_FILENAME,
            repo_type="model", 
            local_dir=".",
            local_dir_use_symlinks=False
        )
        # st.success(f"Model '{HF_MODEL_FILENAME}' successfully loaded from {HF_MODEL_REPO_ID}!", icon="📦")
        # Suppress this successful message after the app is styled
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}", icon="⚠️")
        st.stop() 

# --- Streamlit Application Layout ---

st.title("⚙️ Predictive Engine Maintenance Dashboard")
st.markdown("### Forecast potential engine failures using real-time sensor data.")

# Load the trained model
model = load_model()

if model is not None:
    # --- Input Form for Sensor Readings ---
    
    st.markdown("---")
    st.header("Input Sensor Readings")

    # Dictionary to hold the user inputs
    input_data = {}

    # Define the input columns in a three-column layout
    col1, col2, col3 = st.columns(3)

    # Column 1: Speed and Pressure 1
    with col1:
        st.markdown("#### Engine Speed")
        # Engine_RPM: Range from EDA was approx 61 to 2239
        input_data['Engine_RPM'] = st.number_input(
            "RPM (Revolutions per Minute)",
            min_value=60, max_value=2500, value=790, step=10, 
            key="rpm_input", help="Typical operating speed is 750-850 RPM."
        )
        st.markdown("#### Oil & Fuel Pressures")
        # Lub_Oil_Pressure: Range was approx 0.003 to 7.26
        input_data['Lub_Oil_Pressure'] = st.number_input(
            "Lub Oil Pressure (bar)",
            min_value=0.0, max_value=8.0, value=3.30, step=0.1, format="%.2f",
            key="oil_pressure_input", help="Pressure of the lubricating oil system."
        )
    
    # Column 2: Pressures 2
    with col2:
        st.markdown("#### Fuel & Coolant Pressures")
        # Fuel_Pressure: Range was approx 0.003 to 21.13
        input_data['Fuel_Pressure'] = st.number_input(
            "Fuel Pressure (bar)",
            min_value=0.0, max_value=25.0, value=6.60, step=0.1, format="%.2f",
            key="fuel_pressure_input", help="Pressure applied to deliver fuel to the engine."
        )
        # Coolant_Pressure: Range was approx 0.002 to 7.47
        input_data['Coolant_Pressure'] = st.number_input(
            "Coolant Pressure (bar)",
            min_value=0.0, max_value=8.0, value=2.30, step=0.1, format="%.2f",
            key="coolant_pressure_input", help="Pressure within the engine cooling system."
        )
    
    # Column 3: Temperatures
    with col3:
        st.markdown("#### Temperatures (°C)")
        # Lub_Oil_Temperature: Range was approx 71 to 89
        input_data['Lub_Oil_Temperature'] = st.number_input(
            "Lub Oil Temperature (°C)",
            min_value=70.0, max_value=100.0, value=78.0, step=0.1, format="%.2f",
            key="oil_temp_input", help="Temperature of the circulating lubricating oil."
        )
        # Coolant_Temperature: Range was approx 71 to 102
        input_data['Coolant_Temperature'] = st.number_input(
            "Coolant Temperature (°C)",
            min_value=70.0, max_value=110.0, value=78.0, step=0.1, format="%.2f",
            key="coolant_temp_input", help="Temperature of the engine coolant."
        )
    
    st.markdown("---")
    
    # --- Prediction Logic ---
    col_pred_btn, col_spacer = st.columns([1, 4])
    with col_pred_btn:
        if st.button("Predict Engine Condition", type="primary", use_container_width=True):
            # 1. Get the inputs and save them into a dataframe
            input_df = pd.DataFrame([input_data])

            # 2. Ensure the order of columns matches the training data (CRITICAL)
            input_df = input_df[INPUT_COLUMNS]

            # 3. Make Prediction
            try:
                # Predict probability for both classes (0 and 1)
                prediction_proba = model.predict_proba(input_df)[0]
                # Prediction is the class index (0 or 1)
                prediction = model.predict(input_df)[0]

                # 4. Display Result
                st.session_state['prediction'] = prediction
                st.session_state['proba_success'] = prediction_proba[0]*100
                st.session_state['proba_failure'] = prediction_proba[1]*100
                st.session_state['input_df'] = input_df

            except Exception as e:
                st.error(f"An error occurred during prediction. Full error: {e}")

    # --- Display Result Section ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Analysis & Status")
    
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        proba_success = st.session_state['proba_success']
        proba_failure = st.session_state['proba_failure']
        input_df = st.session_state['input_df']
        
        # Use a container for a clean result card
        result_container = st.container()

        if prediction == 1:
            with result_container:
                st.markdown('<div class="result-card result-card-failure">', unsafe_allow_html=True)
                st.markdown("## 🚨 FAULT PREDICTED - ACTION REQUIRED")
                
                col_status, col_details = st.columns([1, 2])
                with col_status:
                    st.metric(label="Risk of Failure", value=f"{proba_failure:.2f}%", delta="High Risk", delta_color="inverse")
                with col_details:
                    st.warning("Immediate inspection and preventive maintenance are **strongly recommended** to avoid unexpected breakdown, costly repairs, and operational downtime.", icon="🛠️")
                
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            with result_container:
                st.markdown('<div class="result-card result-card-success">', unsafe_allow_html=True)
                st.markdown("## ✅ NORMAL OPERATION - ALL CLEAR")
                
                col_status, col_details = st.columns([1, 2])
                with col_status:
                    st.metric(label="Confidence in Normalcy", value=f"{proba_success:.2f}%", delta="Low Risk", delta_color="normal")
                with col_details:
                    st.info("The engine is operating within normal parameters. Continue with scheduled monitoring and maintenance protocol.", icon="👍")
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Show the data that was fed to the model in an expander
        with st.expander("View Sensor Data Used for Prediction"):
            st.dataframe(input_df, hide_index=True, use_container_width=True)

    else:
        st.info("Click the 'Predict Engine Condition' button above to run the analysis.")

else:
    st.warning("Cannot proceed without a successfully loaded model. Please ensure the model exists in the Hugging Face repo.")
