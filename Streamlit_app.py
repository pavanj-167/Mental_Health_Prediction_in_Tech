import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from data_preprocessing import preprocess_data
from model_training import train_models
from model_inferencing import inference_models
from data_visualization import plot_missing_data,plot_treatment_vs_work_interfere,plot_age_distribution,plot_correlation,plot_target_correlation,plot_treatment_distribution
from model_prediction import prediction

def main():
    st.title("Data Analysis & Prediction for mental health in tech industry")
    
    menu = ["Data Preprocessing", "Data Visualization", "Model Training", "Model Inferencing", "Model Prediction"]
    
    st.sidebar.markdown(
        """
        <style>
        .sidebar-heading {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        </style>
        <div class="sidebar-heading">Contents</div>
        """,
        unsafe_allow_html=True
    )
    
    choice = st.sidebar.selectbox(
        "",
        menu,
        format_func=lambda x: f"🟢 {x}"  
    )
    if choice == "Data Preprocessing":
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv"], key="preprocess_uploader")
        if uploaded_file:
            df = preprocess_data(uploaded_file)
            st.write(df.head())
    elif choice == "Data Visualization":
        df = pd.read_csv(r'/workspaces/Mental_Health_Prediction_in_Tech/datasets/survey.csv')
        plot_missing_data(df)
        df=pd.read_csv(r'/workspaces/Mental_Health_Prediction_in_Tech/datasets/training_data.csv')
        plot_correlation(df)
        plot_treatment_vs_work_interfere(df),plot_age_distribution(df),plot_target_correlation(df,"treatment"),
        plot_treatment_distribution(df)
    elif choice == "Model Training":
        uploaded_file = st.file_uploader("Upload your dataset for training", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            train_models(df)
    elif choice == "Model Inferencing":
        uploaded_file = st.file_uploader("Upload your dataset for inferencing", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            inference_models(df)
    elif choice == "Model Prediction":
        prediction()

if __name__ == "__main__":
    main()
