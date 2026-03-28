import streamlit as st

st.set_page_config(page_title="ML Explanation", page_icon="📖")

st.title("Machine Learning Ensemble Model")

st.header("1. Dataset Overview")
st.markdown("""
**Bank Marketing Dataset**
- **Source**: UCI Machine Learning Repository
- **Task**: Predict whether a client will subscribe to a term deposit (binary classification).
- **Features**: Includes age, job, marital status, education, default, balance, housing, loan, contact type, etc.
""")

st.header("2. Data Preparation & Imperfections")
st.markdown("""
Real-world data is rarely perfect. This dataset contains several imperfections:
- **'Unknown' values**: Several categorical columns (`job`, `marital`, `education`, `default`, `housing`, `loan`) contain 'unknown' as a category.
- **Class Imbalance**: The target variable `y` ('yes'/'no' for term deposit) is highly skewed towards 'no'.

**Preparation Steps**:
1. **Imputation**: We used a `SimpleImputer` to ensure any missing values are handled robustly (medians for numericals, constants for categoricals).
2. **Encoding**: Categorical features were one-hot encoded (`OneHotEncoder`) to convert them to numerical format suitable for ML algorithms.
3. **Scaling**: Numerical features were standardized (`StandardScaler`).
""")

st.header("3. Model Development")
st.markdown("""
We built an **Ensemble Model** using a `VotingClassifier` (soft voting), which averages the predicted probabilities of 3 distinct base models:
1. **Random Forest**: An ensemble of decision trees that improves predictive accuracy and controls over-fitting.
2. **Gradient Boosting**: Builds trees sequentially, where each tree corrects the errors of the previous ones.
3. **Logistic Regression**: A linear model for classification that models the probability of the default class.

**Why Ensemble?**
Combining multiple models often yields better generalization and robustness compared to a single model.
""")
