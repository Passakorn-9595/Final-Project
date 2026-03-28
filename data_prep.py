import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing

def load_bank_marketing():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    os.makedirs("data", exist_ok=True)
    zip_path = "data/bank.zip"
    
    if not os.path.exists(zip_path):
        print("Downloading Bank Marketing Dataset...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/bank")
        print("Downloaded and extracted.")
        
    df = pd.read_csv("data/bank/bank-full.csv", sep=";")
    return df

def get_bank_preprocessing_pipeline(df):
    X = df.drop('y', axis=1)
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor

def load_california_housing():
    print("Loading California Housing Dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Introduce some missing values artificially to AveBedrms to simulate the original dataset's total_bedrooms missing values
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, 'AveBedrms'] = np.nan
    return df

if __name__ == "__main__":
    df_bank = load_bank_marketing()
    print("Bank Marketing Data loaded. Shape:", df_bank.shape)
    
    df_cali = load_california_housing()
    print("California Housing Data loaded. Shape:", df_cali.shape)
    print("California Housing Missing values in AveBedrms:", df_cali['AveBedrms'].isnull().sum())
