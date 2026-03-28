import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import os

st.set_page_config(page_title="NN Testing", page_icon="🧪")

st.title("Test the Neural Network")

class CaliforniaHousingNN(nn.Module):
    def __init__(self, input_dim):
        super(CaliforniaHousingNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

@st.cache_resource
def load_nn_model_and_artifacts():
    imputer = joblib.load('models/nn_imputer.pkl')
    scaler = joblib.load('models/nn_scaler.pkl')
    
    model = CaliforniaHousingNN(input_dim=8)
    model.load_state_dict(torch.load('models/nn_weights.pth', weights_only=True))
    model.eval()
    
    return model, imputer, scaler

try:
    if not os.path.exists('models/nn_weights.pth'):
        st.warning("Training Neural Network... Please wait a moment for 'train_nn.py' to finish in the background, then refresh.")
        st.stop()
        
    model, imputer, scaler = load_nn_model_and_artifacts()
    st.success("Neural Network and preprocessing artifacts loaded successfully!")
    
    st.markdown("### Input District Information")
    st.info("Adjust the sliders to predict the median house value for a California district.")
    
    col1, col2 = st.columns(2)
    with col1:
        med_inc = st.number_input("Median Income (tens of thousands)", min_value=0.5, max_value=15.0, value=3.8, step=0.1)
        house_age = st.number_input("House Median Age", min_value=1.0, max_value=52.0, value=28.0, step=1.0)
        ave_rooms = st.number_input("Average Rooms", min_value=1.0, max_value=10.0, value=5.4, step=0.1)
        ave_bedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    
    with col2:
        population = st.number_input("Population", min_value=10.0, max_value=10000.0, value=1400.0, step=10.0)
        ave_occup = st.number_input("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
        latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.6, step=0.1)
        longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-119.5, step=0.1)
        
    if st.button("Predict House Value"):
        input_data = pd.DataFrame([[
            med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude
        ]], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])
        
        # Preprocess
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            prediction = model(input_tensor).item()
            
        st.success(f"### Predicted Median House Value: ${prediction * 100000:,.0f}")
        
except Exception as e:
    st.error(f"Failed to load the model. Ensure you have run 'train_nn.py' first. Error: {e}")
