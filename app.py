import streamlit as st

st.set_page_config(
    page_title="Data Science Final Project",
    page_icon="🔬",
    layout="wide",
)

st.title("Project IS 2568: Data Science Final Project")

st.markdown("""
### Welcome to the Final Project Showcase!
This web application demonstrates the end-to-end data science process on two distinct datasets.
            
We have developed two models:
1. **Machine Learning Ensemble Model**: A voting classifier combining Random Forest, Gradient Boosting, and Logistic Regression on the **Bank Marketing Dataset**.
2. **Neural Network**: A custom feed-forward neural network predicting housing prices on the **California Housing Dataset**.

Please use the sidebar to navigate to the detailed explanation and interactive testing pages for both models.
""")

st.info("Developed for Project IS 2568.")

st.markdown("---")
st.markdown("### Team Members")
st.markdown("""
1. **6604062663248** นายภาสกร ปั้นกาญจนโต sec 5  
2. **6604062663311** นายศุภณัฐ ลองจำนงค์ sec 5
""")
