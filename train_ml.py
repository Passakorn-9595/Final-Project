import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from data_prep import load_bank_marketing, get_bank_preprocessing_pipeline

def train_ensemble():
    print("Loading data...")
    df = load_bank_marketing()
    
    # Map target to binary
    df['y'] = df['y'].map({'no': 0, 'yes': 1})
    
    X = df.drop('y', axis=1)
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Setting up preprocessing and models...")
    preprocessor = get_bank_preprocessing_pipeline(df)
    
    # Define 3 base models
    clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=50, random_state=42)
    clf3 = LogisticRegression(max_iter=1000, random_state=42)
    
    # Combine into a Voting Classifier
    ensemble = VotingClassifier(
        estimators=[('rf', clf1), ('gb', clf2), ('lr', clf3)],
        voting='soft'
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])
    
    print("Training the ensemble model (this might take a minute)...")
    model_pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, 'models/ensemble_bank.pkl', compress=3)
    print("Model saved to models/ensemble_bank.pkl")

if __name__ == "__main__":
    train_ensemble()
