import streamlit as st

st.set_page_config(page_title="NN Explanation", page_icon="📖")

st.title("Neural Network Model")

st.header("1. Dataset Overview")
st.markdown("""
**California Housing Prices**
- **Source**: StatLib repository / 1990 California Census (via Scikit-Learn `fetch_california_housing`).
- **Task**: Predict the median house value for California districts (Regression).
- **Features**: Includes median income, housing median age, total rooms, total bedrooms, population, households, latitude, and longitude.
""")

st.header("2. Data Preparation & Imperfections")
st.markdown("""
- **Missing Values**: We introduced missing values in the `AveBedrms` (Average Bedrooms) column to simulate real-world data imperfections.
- **Handling**: Used `SimpleImputer` (median strategy) to fill missing values before passing to the model.
- **Scaling**: Neural networks are extremely sensitive to unscaled inputs. We applied `StandardScaler` to ensure all numerical features have a mean of 0 and a standard deviation of 1.
""")

st.header("3. Network Architecture")
st.markdown("""
We designed a custom Feed-Forward Neural Network using **PyTorch**:
- **Input Layer**: Takes the 8 scaled features.
- **Hidden Layer 1**: 64 neurons with ReLU activation.
- **Hidden Layer 2**: 32 neurons with ReLU activation.
- **Output Layer**: 1 neuron (linear output for continuous value regression).
- **Optimization**: Trained using the Adam optimizer and Mean Squared Error (MSE) loss function for 50 epochs.
""")
