import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, precision_score

st.title("Sepsis Prediction: Lactate, Temperature & MAP Effect Dashboard")

@st.cache_data
def load_data():
    # Try different possible paths for the CSV file
    csv_paths = [
        "sepsis.csv",  # If file is in the same directory
        r"C:\Users\timsi\OneDrive\Bewaji HealthCare Solutions\training\JAIA AI Masterclass in Health\sepsis.csv",  # Local path
        "data/sepsis.csv",  # If in a data subdirectory
    ]
    
    df = None
    for path in csv_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        except:
            continue
    
    if df is None:
        st.error("Could not load the sepsis.csv file. Please ensure the file is available.")
        return None
    
    df_majority = df[df.SepsisLabel==0]
    df_minority = df[df.SepsisLabel==1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

# Load data
df = load_data()
if df is None:
    st.stop()

# Load model
try:
    clf = joblib.load("sepsis_mlp_model.joblib")
except:
    st.error("Could not load the trained model file (sepsis_mlp_model.joblib). Please ensure the model is trained and saved.")
    st.stop()

# Find column names
lactate_col = 'Lactate'  # Change if your column is named differently
temp_col = 'Temp'        # Change if your column is named differently
map_col = 'MAP'          # Change if your column is named differently
lactate_min = float(df[lactate_col].min())
lactate_max = float(df[lactate_col].max())
lactate_mean = float(df[lactate_col].mean())
temp_mean = float(df[temp_col].mean())

# Sliders
lactate_value = st.slider("Lactate value", min_value=lactate_min, max_value=lactate_max, value=lactate_mean, step=0.1)
temp_value = st.slider("Temperature (°C)", min_value=30.0, max_value=45.0, value=temp_mean, step=0.1)
map_value = st.slider("MAP (mm Hg)", min_value=1, max_value=200, value=70, step=1)
threshold = st.slider("Sepsis probability threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Prepare a sample for prediction (user-set lactate, temperature, MAP)
feature_cols = df.columns[:-1]  # Exclude SepsisLabel
sample = df[feature_cols].mean().values.reshape(1, -1)
lactate_idx = list(feature_cols).index(lactate_col)
temp_idx = list(feature_cols).index(temp_col)
map_idx = list(feature_cols).index(map_col)
sample[0, lactate_idx] = lactate_value
sample[0, temp_idx] = temp_value
sample[0, map_idx] = map_value

# Predict probability for user-set values
prob = clf.predict_proba(sample)[0, 1]

# Display probability and Yes/No
st.write(f"**Predicted sepsis probability (for lactate {lactate_value:.1f}, temp {temp_value:.1f}°C, MAP {map_value} mm Hg):** {prob:.3f}")
if prob >= threshold:
    st.success("YES: Likely to have sepsis")
else:
    st.info("NO: Not likely to have sepsis")

st.write(f"**Current threshold:** {threshold:.2f}")

# --- Alerts per hour for the whole test set at current threshold ---

# Prepare test set as in training
X = df[feature_cols].values
Y = df['SepsisLabel'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Get the corresponding test set DataFrame
_, test_indices = train_test_split(np.arange(len(df)), test_size=0.20, random_state=0)
test_df = df.iloc[test_indices]

# Predict probabilities for test set
probs_test = clf.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= threshold).astype(int)

test_df = test_df.copy()
test_df['PredictedAlert'] = preds_test

# Calculate total ICULOS hours in the test set
if 'ICULOS' in test_df.columns:
    total_iculos_hours = test_df['ICULOS'].sum()
else:
    total_iculos_hours = 0

# Calculate number of predicted alerts
num_alerts = test_df['PredictedAlert'].sum()

# Calculate alerts per hour
alerts_per_hour = num_alerts / total_iculos_hours if total_iculos_hours > 0 else 0

st.write(f"**Predicted alerts per hour of ICU stay (whole test set, at threshold {threshold:.2f}):** {alerts_per_hour:.4f}") 