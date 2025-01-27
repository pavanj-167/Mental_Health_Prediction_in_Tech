import joblib
import pandas as pd
import streamlit as st 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def inference_models(input_file):
    model_lr = joblib.load(r"/workspaces/Mental_Health_Prediction_in_Tech/models/logistic_regression_model.pkl")
    model_rf = joblib.load(r"/workspaces/Mental_Health_Prediction_in_Tech/models/random_forest_model.pkl")
    scaler = joblib.load(r"/workspaces/Mental_Health_Prediction_in_Tech/models/scaler.pkl")
    
    st.subheader("Model Inference")
    X_test= input_file.drop(columns=["treatment"])
    y_test =input_file['treatment']
    
    scaler=StandardScaler()
    X_test_scaled=scaler.fit_transform(X_test)

    # Prediction for logistic Regression
    y_pred_lr = model_lr.predict(X_test_scaled)
    st.subheader("Classification Report for Logistic Regression")
    lr_report= classification_report(y_test, y_pred_lr, output_dict=True)  
    lr_report_df = pd.DataFrame(lr_report).transpose()  
    st.table(lr_report_df)
    st.write("Inference Accuracy for Logistic Regression:", accuracy_score(y_test, y_pred_lr)) 
    st.subheader("Confusion Matrix for Logistic Regression")
    st.write(confusion_matrix(y_test,y_pred_lr))

    # Predicition for random forest 
    y_pred_rf = model_rf.predict(X_test)
    st.subheader("Classification Report for Random Forest")
    rf_report = classification_report(y_test, y_pred_rf, output_dict=True)  
    rf_report_df = pd.DataFrame(rf_report).transpose() 
    st.table(rf_report_df)  
    st.write("Inference Accuracy for Random Forest:", accuracy_score(y_test, y_pred_rf)) 
    st.subheader("Confusion Matrix for Random Forest Classifier")
    st.write(confusion_matrix(y_test,y_pred_rf))
    return y_pred_lr, y_pred_rf

