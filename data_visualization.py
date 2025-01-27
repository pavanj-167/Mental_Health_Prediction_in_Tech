import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data
def plot_missing_data(df):
    st.subheader("Missing data HeatMap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    st.pyplot(plt)

def plot_treatment_vs_work_interfere(df):
    st.subheader("Treatment Based on Work Interfere")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='work_interfere', hue='treatment', data=df)
    st.pyplot(plt)

def plot_age_distribution(df):
    st.subheader("Distribution and Density by Age")
    plt.figure(figsize=(10,8))
    sns.displot(df["Age"], bins=24, kde=True)
    plt.title("Age Distribution and Density")
    st.pyplot(plt)

def plot_correlation(df):
    st.subheader("Correlation HeatMap")
    correlation = df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.show() 
    st.pyplot(plt)

def plot_target_correlation(df,treatment):
    st.subheader("Correlation with Target Variable")
    target_corr = df.corr()[treatment].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=target_corr.index, y=target_corr.values)
    plt.xticks(rotation=90)
    plt.title("Correlation with Target Variable")
    st.pyplot(plt)

def plot_treatment_distribution(df):
    st.subheader("Total Distribution by Treatment")
    plt.figure(figsize=(10, 3))
    sns.countplot(y="treatment", hue="treatment", data=df, palette='Set2', dodge=False)
    plt.title('Total Distribution by Treatment')
    st.pyplot(plt)

