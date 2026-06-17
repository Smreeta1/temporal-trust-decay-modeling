
import sys
import os

# find the absolute path of the parent directory and add it to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib.pyplot as plt
from src.data_generator import generate_synthetic_data
from src.trust_model import TrustDecayModel

st.set_page_config(page_title="Trust Decay Modeler", layout="wide")

st.title("Dynamic Trust-Decay Temporal Modeling")
st.markdown("A research pipeline tracking user velocity and NLP stance shifts to detect misinformation spreaders.")

# Sidebar Controls for the Professor to play with
st.sidebar.header("Model Hyperparameters")
lambda_val = st.sidebar.slider("Retention Factor (Lambda)", 0.0, 1.0, 0.95)
gamma_val = st.sidebar.slider("Decay Multiplier (Gamma)", 0.0, 5.0, 2.0)

# 1. Load Data
st.subheader("1. Data Ingestion (Simulated User Stream)")
df_raw = generate_synthetic_data(100)
st.dataframe(df_raw.tail(5))

# 2. Process Model
model = TrustDecayModel(lambda_factor=lambda_val, gamma=gamma_val)
df_processed = model.process_pipeline(df_raw)

# 3. Visualize Results
st.subheader("2. Temporal Trust Decay Analysis")

fig, ax1 = plt.subplots(figsize=(10, 4))

# Plot Trust Score
color = 'tab:blue'
ax1.set_xlabel('Timeline (Tweets)')
ax1.set_ylabel('Trust Score', color=color)
ax1.plot(df_processed.index, df_processed['trust_score'], color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(y=30, color='r', linestyle='--', label='Suspension Threshold')

# Plot Velocity on second Y axis
ax2 = ax1.twinx()
color = 'tab:gray'
ax2.set_ylabel('Sharing Velocity', color=color)
ax2.bar(df_processed.index, df_processed['velocity'], color=color, alpha=0.3)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
st.pyplot(fig)

st.markdown("""
**Observation:** You can observe the trust score remain stable during normal sharing behavior. 
When the user's temporal velocity increases rapidly alongside highly polarized text, the trust score decays exponentially, falling below the red threshold.
""")