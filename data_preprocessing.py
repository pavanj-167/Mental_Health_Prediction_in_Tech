import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
def preprocess_data(input_file):
    df = pd.read_csv(input_file)
    st.write(df.head())
    st.subheader("After dropping some columns")
    columns_to_drop = ['Timestamp', 'Country', 'state', 'comments']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
    st.write(df.head()) 

    st.subheader("Dataset Description")
    st.write(df.describe())
    if 'self_employed' in df.columns:
        df['self_employed'] = df['self_employed'].fillna(df['self_employed'].mode()[0])
    if 'work_interfere' in df.columns:
        df['work_interfere'] = df['work_interfere'].fillna(df['work_interfere'].mode()[0])

    if 'Age' in df.columns:
        df['Age'] = df['Age'].apply(lambda x: np.nan if x < 0 or x > 120 else x)
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    if 'Gender' in df.columns:
        gender_map = {'male': ['male ', 'male', 'm', 'cis male', 'man', 'msle'], 
                      'female': ['female ', 'female', 'f', 'woman', 'femail'], 
                      'trans': ['queer', 'non-binary', 'androgyne']}
        for key, values in gender_map.items():
            df['Gender'] = df['Gender'].replace(values, key)
    st.subheader("After Encoding categorical Features")
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df