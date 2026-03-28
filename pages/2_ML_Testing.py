import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ML Testing", page_icon="🧪")

st.title("Test the Ensemble Model")

@st.cache_resource
def load_model_and_data():
    model = joblib.load('models/ensemble_bank.pkl')
    # Load sample to get all columns
    df = pd.read_csv('data/bank/bank-full.csv', sep=';')
    sample = df.drop('y', axis=1).iloc[0].to_dict()
    return model, sample

try:
    model, sample_data = load_model_and_data()
    st.success("Model loaded successfully!")
    
    st.markdown("### Input Client Information")
    st.info("We provide a default client profile. Change the values below to see how the prediction changes!")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=int(sample_data['age']))
        job_options = ['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown']
        job_idx = job_options.index(sample_data['job']) if sample_data['job'] in job_options else 0
        job = st.selectbox("Job", job_options, index=job_idx)
        balance = st.number_input("Yearly Balance (euros)", value=int(sample_data['balance']))
    
    with col2:
        housing = st.selectbox("Has Housing Loan?", ['yes', 'no'], index=0 if sample_data['housing'] == 'yes' else 1)
        loan = st.selectbox("Has Personal Loan?", ['yes', 'no'], index=0 if sample_data['loan'] == 'yes' else 1)
        duration = st.number_input("Last Contact Duration (seconds)", value=int(sample_data['duration']))
        
    if st.button("Predict Subscription"):
        input_data = sample_data.copy()
        input_data['age'] = age
        input_data['job'] = job
        input_data['balance'] = balance
        input_data['housing'] = housing
        input_data['loan'] = loan
        input_data['duration'] = duration
        
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        if prediction == 1:
            st.success(f"The model predicts: **YES** (The client will subscribe) with {prob:.1%} probability.")
        else:
            st.error(f"The model predicts: **NO** (The client will NOT subscribe) with {1-prob:.1%} probability.")
            
except Exception as e:
    st.error(f"Failed to load model or data. Ensure you have run 'train_ml.py' first. Error: {e}")
