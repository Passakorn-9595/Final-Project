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
        med_inc = st.slider("Median Income (tens of thousands)", 0.5, 15.0, 3.8)
        house_age = st.slider("House Median Age", 1.0, 52.0, 28.0)
        ave_rooms = st.slider("Average Rooms", 1.0, 10.0, 5.4)
        ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
    
    with col2:
        population = st.slider("Population", 10.0, 10000.0, 1400.0)
        ave_occup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)
        latitude = st.slider("Latitude", 32.0, 42.0, 35.6)
        longitude = st.slider("Longitude", -124.0, -114.0, -119.5)
        
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
