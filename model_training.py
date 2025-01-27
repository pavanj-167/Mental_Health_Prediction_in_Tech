import pandas as pd
import streamlit as st 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import joblib

def train_models(input_file):
    
    X_train = input_file.drop(columns=["treatment"])
    y_train = input_file["treatment"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Logistic Regression model
    model_lr = LogisticRegression(random_state=43)
    model_lr.fit(X_train_scaled, y_train) 
    y_pred_lr=model_lr.predict(X_train_scaled) 
    st.subheader("Classification Report for Logistic Regression")
    lr_report= classification_report(y_train, y_pred_lr, output_dict=True)  
    lr_report_df = pd.DataFrame(lr_report).transpose()  
    st.table(lr_report_df)
    st.write("Accuracy for Logistic Regression:", accuracy_score(y_train, y_pred_lr)) 
    st.subheader("Confusion Matrix for Logistic Regression")
    st.write(confusion_matrix(y_train,y_pred_lr))

    # Train Random Forest Classifier model
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train_scaled, y_train)

    y_pred_rf=model_rf.predict(X_train)
    st.subheader("Classification Report for Random Forest")
    rf_report = classification_report(y_train, y_pred_rf, output_dict=True)  
    rf_report_df = pd.DataFrame(rf_report).transpose() 
    st.table(rf_report_df)  
    st.write("Accuracy for Random Forest:", accuracy_score(y_train, y_pred_rf)) 
    st.subheader("Confusion Matrix for Random Forest Classifier")
    st.write(confusion_matrix(y_train,y_pred_rf))

    # Saving the model
    joblib.dump(model_lr, "/workspaces/Mental_Health_Prediction_in_Tech/models/logistic_regression_model.pkl")
    joblib.dump(model_rf, "/workspaces/Mental_Health_Prediction_in_Tech/models/random_forest_model.pkl")
    joblib.dump(scaler, "/workspaces/Mental_Health_Prediction_in_Tech/models/scaler.pkl")
    
    return model_lr, model_rf, scaler