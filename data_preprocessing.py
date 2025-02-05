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









# Standard library imports
import os
import re
import time
import uuid
import glob
import shutil
import datetime
from datetime import datetime
from dateutil import tz
# Flask-related imports
from flask import Flask, jsonify, request, send_file,g
from flask_cors import CORS
from werkzeug.utils import secure_filename
# Data manipulation and computation
import numpy as np
import pandas as pd
# Machine learning and preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
# XGBoost
from xgboost import XGBClassifier
# Deep learning (Keras and TensorFlow)
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras import Sequential
from keras.utils import to_categorical
import tensorflow_hub as hub
# Text similarity and natural language processing
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from datasketch import MinHash, MinHashLSH
import spacy
# Transformers
from transformers import pipeline
# Cloud services and configuration
import boto3
import io
import google.generativeai as genai
# Job scheduling and server
import schedule
import threading
from waitress import serve
# Serialization
import joblib
from difflib import SequenceMatcher
import heapq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_google_vertexai import VertexAI
from database_connection import upsert_s3_config,get_s3_connection
load_dotenv()
def load_models(model_prefixes, files_to_avoid):
    loaded_models = {}
    for prefix in model_prefixes:
        for filename in os.listdir('Models'):
            if filename.startswith(prefix) and filename.endswith('.pkl') and filename not in files_to_avoid:
                model_path = os.path.join('Models', filename)
                loaded_models[prefix] = joblib.load(model_path)
                break  # Load only the first model with the specified prefix
    return loaded_models
# Define model prefixes and files to avoid
model_prefixes = ['trained_model', 'stacked_classifier', 'tfidf_vectorizer']
files_to_avoid = ['tfidf_vectorizer_xgb_priority.pkl', 'tfidf_vectorizer_xgboost.pkl']
# Load models
loaded_models = load_models(model_prefixes, files_to_avoid)
retry_count = 0
allowed_retry = 3
# Access the API keys from environment variables
genai_api_key = os.getenv('GENAI_API_KEY')
# Load configuration from config.py
# Initialize the clients with the API keys
genai.configure(api_key=genai_api_key)
# Function to load models
xgboost_model = joblib.load('Models/xgboost.pkl')
tfidvec_xgboost_model = joblib.load('Models/tfidf_vectorizer_xgboost.pkl')
label_encoder_model = joblib.load('Models/label_encoder.pkl')
cnn_label_encoder_model = joblib.load('Models/cnn_label_encoder.pkl')
resolution_model = pd.read_csv('Models/vector_database.csv')
stacked_model = loaded_models.get('stacked_classifier')
svm_model = loaded_models.get('trained_model')
tfid_vec = loaded_models.get('tfidf_vectorizer')
tfidf_vectorizer_priority = joblib.load('Models/tfidf_vectorizer_xgb_priority.pkl')
xgb_classifier_priority = joblib.load('Models/XgBoost_Classier_priority.pkl')
label_encoder_priority = joblib.load('Models/label_encodr_xgb_priority.pkl')
t_hub = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
k_shingle_size = 5  
num_permutations = 256  
model_threshold = 0.85
# Get Resolution from Gemeni
def resolution_llm_gemeni(error_message):
    prompt_message="""
    Instructions: Answer directly without any preamble or explanations.
    Analyze and provide a concise, actionable resolution for the error {error_message} by following 
    these guidelines:
        Begin with identifying the root cause of the error
        Present a clear, sequential troubleshooting process using numbered steps
        Include specific commands, code snippets, or configuration changes where applicable
        Mention any prerequisites or system requirements needed
        Add common pitfalls to avoid during implementation
        Conclude with a verification step to confirm the error is resolved
        If multiple solutions exist, list them in order of simplicity and effectiveness
        Ensure the resolution is practical, directly addresses the error, and maintains brevity 
    while being comprehensive. Focus on essential information without technical jargon, unless
    necessary. Include any relevant documentation references or official resources for further 
    reading. make sure you are sending a proper resolution as a response.
    make sure your resolution does not exceed 50 words. summerize the resolution ands steps to 50 words only
    Answer:
        Resolution:
    """
    llm = VertexAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    max_retries=5,  
    retry_on_timeout=True,  
    )
    output = llm.invoke(prompt_message.format(error_message=error_message))
    return [output.replace('*','')]
def pre_preocess_errors(error_message):
    error_message = error_message.strip()
    # Define a regex pattern to match and replace other numbers (not common error codes)
    error_message = re.sub(r'(?<!\d)(?<!404)(?<!500)(?<!401)(?<!403)\d(?!04)(?!00)(?!01)(?!03)\d*(?!\d)', '', error_message)
    # Remove email addresses, URLs, and other characters
    error_message = re.sub(r'https?://\S+', '', error_message)
    error_message = re.sub(r'\S+@\S+', '', error_message)
    error_message = re.sub(r'[.,:\'"]', '', error_message)
    # Remove extra spaces and return the processed error message
    error_message = ' '.join(error_message.split())
    return error_message
def calculate_similarity(error, resolution,model):
    embeddings = model.encode([error, resolution], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()
def get_best_resolution(resolution,error):
    # Load your CSV data
    start_time = time.time()
    # Load the model with from_tf=True
    model = SentenceTransformer('all-MiniLM-L6-v2')
    best_resolutions = []
    if pd.notnull(resolution):  # Check if resolution is not NaN
        resolutions = resolution.split("\n")  # Assuming resolutions are separated by newline
        max_similarity = -1
        best_resolution = ""
        for res in resolutions:
            res = res.split("\\rmv")[0]
            similarity = calculate_similarity(error, res,model)
            if similarity > max_similarity and len(res)>25:
                max_similarity = similarity
                best_resolution = res.replace("###","\n")
        best_resolutions.append(best_resolution)
    else:
        best_resolutions.append("")
    end_time = time.time()
    time_to_execute = end_time - start_time
    print(f"To get the best resolution : {time_to_execute} sec")
    return best_resolutions
def trian_resolution_model():
    # Load your CSV data
    df = pd.read_csv('Data/Training/Resolution/resolution.csv')
    df = df.dropna()
    #.apply(pre_preocess_errors)
    # Combine relevant columns into a single text field
    df['Combined_Text'] = df['Error'] + ' ' + df['Resolution']
    df['Combined_Text'] = df['Combined_Text'].apply(pre_preocess_errors)
    # Load Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # Vectorize the combined text
    vectors = embed(df['Combined_Text'].tolist())
    vector_db = pd.DataFrame(vectors.numpy(), columns=[f'vector_{i}' for i in range(vectors.shape[1])])
    vector_db['Resolution'] = df['Resolution']
    vector_db.to_csv('Models/vector_database.csv', index=False) 
#Generating reports
def generate_reports(prediction_data,accurate_data,dataframe):
    accuracy = accuracy_score(accurate_data, prediction_data)
    classification_rep = classification_report(accurate_data, prediction_data)
    conf_matrix = confusion_matrix(accurate_data, prediction_data)
    results_df = pd.DataFrame({
    'Error': dataframe['Error'],
    'True_Isue' : dataframe['Issue'],
    'True_Sub_Issue': dataframe['Sub_Issue'],
    'Predicted_Sub_Issue': prediction_data
    })
    return accuracy, classification_rep, conf_matrix, results_df 
# In[9]:
#Get Predictions for lablel encoded models
def get_predictions_from_csv_encode(csv_file):
    global xgboost_model
    global tfidvec_xgboost_model
    global label_encoder_model
    if xgboost_model is not None and tfidvec_xgboost_model is not None and label_encoder_model is not None:
        #Loading The CSV File to test
        df = pd.read_csv(csv_file)
        #Getting the Feature and Target Values 
        X_test_data = df['Error']
        y_test_data = df['Sub_Issue']
        y_test_data = label_encoder_model.fit_transform(y_test_data)
        #Encoding the Feature Data 
        encoded_test_X = tfidvec_xgboost_model.transform(X_test_data)  
        #Getting Predections
        test_predictions = xgboost_model.predict(encoded_test_X) 
        # Inverse Transform to Decode Predictions
        decoded_predictions = label_encoder_model.inverse_transform(test_predictions)
        #Printing The Reports Remove If not needed
        accuracy,classification_rep,conf_matrix, results_df =   generate_reports(test_predictions,y_test_data,df)
        return decoded_predictions, accuracy, classification_rep, conf_matrix,results_df
    else:
        return ("Load the model and the encoder"), None, None ,None, None
# In[10]:
# Global variable for the heap map
error_heap_map = []
def calculate_similarity_summerization(error1, error2):
    similarity = SequenceMatcher(None, error1, error2).ratio()
    return similarity
def insert_to_heap_map(error):
    global error_heap_map
    for existing_error, existing_summary in error_heap_map:
        similarity = calculate_similarity_summerization(error, existing_error)
        if similarity > 0.8:  
            return existing_summary  
    error=pre_preocess_errors(error)
    new_summary = summarize(error)
    heapq.heappush(error_heap_map, (error, new_summary))
    return new_summary
def load_model_and_vectorizer(model_date):
    model_file = f"Models/old_models/stacked_classifier_{model_date}.pkl"
    vectorizer_file = f"Models/old_models/tfidf_vectorizer_{model_date}.pkl"
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        stacked_model = joblib.load(model_file)
        tfid_vec = joblib.load(vectorizer_file)
        return stacked_model, tfid_vec
    return None, None
def get_predections_csv_stacked(csv_file, model, vectorizer):
    if model is not None and vectorizer is not None:
        df = pd.read_csv(csv_file)
        X_test_data = df['Error'].apply(pre_preocess_errors)
        encoded_test_X = vectorizer.transform(X_test_data)
        test_predictions = model.predict(encoded_test_X)
        new_probability_estimates = model.predict_proba(encoded_test_X)
        results_df = pd.DataFrame({
            'error': df['Error'],
            'sub_issue': test_predictions,
            'issue': None,
            'resolution': None,
            'percentage': round(np.max(new_probability_estimates) * 100, 2),
            'summary': None,
            'priority': None
        })
        for index, row in results_df.iterrows():
            results_df.at[index, 'summary'] = insert_to_heap_map(row['error'])
            results_df.at[index, 'issue'] = get_issue(row['sub_issue'])
            results_df.at[index, 'resolution'] = resolution_llm_gemeni(row['error'])
            results_df.at[index, 'priority'] = predict_priority(row['error'])
        return results_df
    else:
        return "Load the model and the encoder"
def get_predections_csv_xgboost(csv_file):
    global xgboost_model
    global tfidvec_xgboost_model
    if xgboost_model is not None and tfidvec_xgboost_model is not None:
        #Loading The CSV File to test
        df = pd.read_csv(csv_file)
        #Getting the Feature and Target Values 
        X_test_data = df['Error']
        encoded_test_X = tfidvec_xgboost_model.transform(X_test_data)  
        #Getting Predections
        test_predictions = xgboost_model.predict(encoded_test_X) 
        # Inverse Transform to Decode Predictions
        decoded_predictions = label_encoder_model.inverse_transform(test_predictions)
        results_df = pd.DataFrame({
        'error': df['Error'],
        'sub_issue' : decoded_predictions,
        'issue' :None,
        'resolution':None
        })
        for index, row in results_df.iterrows():
            row['issue'] = get_issue(row['sub_issue'])
            row['resolution'] = get_resolution(row['error'])
        return results_df
    else:
        return ("Load the model and the encoder")
def get_predections_csv_stacked_list(df):
    global stacked_model
    global tfid_vec 
    if stacked_model is not None and tfid_vec is not None:
        #Loading The CSV File to test
        #Getting the Feature and Target Values 
        X_test_data = df['Error'].apply(pre_preocess_errors)
        encoded_test_X = tfid_vec.transform(X_test_data)  
        #Getting Predections
        test_predictions = stacked_model.predict(encoded_test_X) 
        # Inverse Transform to Decode Predictions
        new_probability_estimates = stacked_model.predict_proba(encoded_test_X)
        results_df = pd.DataFrame({
        'error': df['Error'],
        'sub_issue' : test_predictions,
        'issue' :None,
        'resolution':None,
        'percentage':round(np.max(new_probability_estimates) * 100, 2),
        'summary': None,
        'priority':None
        })
        for index, row in results_df.iterrows():
            results_df.at[index, 'issue'] = get_issue(row['sub_issue'])
            results_df.at[index, 'resolution'] = get_resolution(row['error'])
            results_df.at[index, 'summary'] = summarize(row['error'])
            results_df.at[index, 'priority'] = predict_priority(row['error'])
        return results_df
    else:
        return ("Load the model and the encoder")
# In[13]:
def get_predections_csv_xgboost_list(df):
    global xgboost_model
    global tfidvec_xgboost_model
    if xgboost_model is not None and tfidvec_xgboost_model is not None:
        #Loading The CSV File to test
        #Getting the Feature and Target Values 
        X_test_data = df['Error']
        encoded_test_X = tfidvec_xgboost_model.transform(X_test_data)  
        #Getting Predections
        test_predictions = xgboost_model.predict(encoded_test_X) 
        # Inverse Transform to Decode Predictions
        decoded_predictions = label_encoder_model.inverse_transform(test_predictions)
        results_df = pd.DataFrame({
        'error': df['Error'],
        'sub_issue' : decoded_predictions,
        'issue' :None,
        'resolution' : None
        })
        for index, row in results_df.iterrows():
            row['issue'] = get_issue(row['sub_issue'])
            row['resolution'] = get_resolution(row['error'])
        return results_df
    else:
        return ("Load the model and the encoder")
# In[14]:
def get_prediction_stacked_value(text):
    start_time = time.time()
    global stacked_model
    global tfid_vec 
    text =pre_preocess_errors(text)
    new_data = [text]
    # Vectorize the new data
    new_data_tfidf = tfid_vec.transform(new_data)
    # Make predictions using the trained Stacking Classifier
    y_pred = stacked_model.predict(new_data_tfidf)
    value  = y_pred[0]
    new_probability_estimates = stacked_model.predict_proba(new_data_tfidf)
    max_value = np.max(new_probability_estimates[0])
    max_percentage = round(max_value * 100, 2)
    end_time = time.time()
    time_to_execte = end_time - start_time
    print(f"Time to get a issue pred {time_to_execte} sec")
    return str(value),max_percentage
# In[15]:
def get_sub_issue_xgboost(custom_text):
    global xgboost_model
    global label_encoder_model
    if xgboost_model is not None and label_encoder_model is not None:
        custom_text = [custom_text]
        custom_text = tfidvec_xgboost_model.transform(custom_text) 
        custom_preds = xgboost_model.predict(custom_text)
        predicted_classes = label_encoder_model.inverse_transform(custom_preds)
        prediction_text = ', '.join(predicted_classes)
        return prediction_text
    return ("Load the model and the encoder")
# In[16]:
dataframe = pd.read_csv("Data/Training/Categorization/train.csv",delimiter="\t")
unique_rows = dataframe.drop_duplicates(subset=["Sub_Issue"])
unique_rows = unique_rows.drop(columns=['Error'])
def get_issue(sub_issue):
    global dataframe
    # Filter the DataFrame based on the given Sub Issue
    filtered_df = dataframe[dataframe['Sub_Issue'] == sub_issue]
    # Check if there are any issues
    if not filtered_df.empty:
        # Assuming there is only one unique issue for the given Sub Issue
        unique_issue = filtered_df['Issue'].unique()[0]
        return unique_issue
    else:
        return f"No issue found for Sub Issue '{sub_issue}'"
# intergrading the parser 
def remove_word_at_start(line, word_to_remove):
    if line.startswith(word_to_remove + ' '):
        line = line[len(word_to_remove) + 1:]
    return line
def remove_non_string_start(line):
    match = re.search(r'\A[^a-zA-Z]*', line)
    if match:
        return line[match.end():]  
    return line
patterns = [
    r'ERROR \d+ --- \[.*?\]',
    r'WARN 1 --- \[.*?\]',
    r'[A-Z][a-zA-Z]*Error\b',
    r'java\.lang\..*Exception.*\n(\s+at .+\n)*',
    r'\b(4\d{2}|5\d{2})\b\s+[A-Za-z\s]+HTTP/\d\.\d',
    r'FileNotFoundError: .*No such file or directory',
    r'DatabaseError: .*',
    r'Permission denied: .*',
    r'Out of memory|java\.lang\.OutOfMemoryError',
    r'TimeoutError: .*',
    r'SEVERE.*',
    r'Error.*(?=\s+\w+:|$)',
    r'ERROR.*(?=\s+\w+:|$)',
    r'.*error\scode\s.*',
    # r"'.*?' to (\w+)",
]
compiled_patterns = [re.compile(pattern) for pattern in patterns]
keywords = ['exception']
exclusion_pattern = re.compile(r'checkin\.source\.exception|checkin\.bff\.exception|checkin\.core\.exception')
def contains_pattern_or_keyword(line):
    # Newly aded following line for testing
    if "HttpClientErrorException" in line:
        return False 
    if "InternalServerError: 500" in line:
        return False
    if exclusion_pattern.search(line):
        return False  # Exclude this line
    for pattern in compiled_patterns:
        if pattern.search(line):
            # print(pattern)
            return True
    for keyword in keywords:
        if keyword.lower() in line.lower():
            return True
    return False
def remove_patterns_from_selected_lines(lines):
    patterns_to_remove = [
        r'ERROR \d+ --- \[.*?\]',
        r'WARN 1 --- \[.*?\]',
        r'DEBUG 1 --- \[.*?\]',
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}', 
        r'^\s*INFO 1 --- \[.*\]\s*',
        r'^(\d+)m(\d{2}:\d{2}:\d{2},\d{3})$',
        r'\x1B\[[0-9;]*[mGK]',
        r'\bWARN\b',
        r'.*c\.n\.eureka\.util\.batcher\.TaskExecutors.*',
        r'CHECKIN-CLOUD-GATEWAY,,\] 1 --- \[or-http-epoll-(\d+)\]',
        r'\[([^]]+)\]',
    ]
    compiled_patterns = [re.compile(pattern) for pattern in patterns_to_remove]
    cleaned_lines = set()
    for line in lines:
        cleaned_line = line.strip() 
        for pattern in compiled_patterns:
            cleaned_line = pattern.sub('', cleaned_line)
        cleaned_line= cleaned_line.encode('ascii', 'ignore').decode('ascii')
        cleaned_line = re.sub(r'[^\x20-\x7E]+', '', cleaned_line)
        cleaned_line = remove_non_string_start(cleaned_line) 
        cleaned_line = cleaned_line.replace('"', '')
        cleaned_lines.add(cleaned_line)
    return cleaned_lines
def process_log(input_file_name):
    start_time = time.time()
    selected_lines = []
    try:
        with open(input_file_name, "r") as input_file:
            for line in input_file:
                if contains_pattern_or_keyword(line):
                    print("the selected lines are",line)
                    selected_lines.append(line)
    except UnicodeDecodeError:
        # Retry opening the file with utf-8 encoding in case of a decoding error
        with open(input_file_name, "r", encoding='iso-8859-1') as input_file:
            for line in input_file:
                if contains_pattern_or_keyword(line):
                    print("the selected lines are",line)
                    selected_lines.append(line)
    cleaned_lines = remove_patterns_from_selected_lines(selected_lines)
    modified_lines = set()
    for line in cleaned_lines:
        if '.' in line:  
            line = remove_word_at_start(line, "at")
            line = remove_word_at_start(line, "Caused by:")
            line = line.strip()
            modified_lines.add(line)
    df = pd.DataFrame(modified_lines, columns=["Error"])
    unique_filename = str(uuid.uuid4())
    file = "csv_files"
    full_file_name = file + "/" + unique_filename + ".csv"
    if not os.path.exists(file):
        os.makedirs(file)
    df.to_csv(full_file_name, index=False)
    end_time = time.time()
    time_to_execte = end_time - start_time
    print(f"Time to parse {time_to_execte} sec")
    return full_file_name
def move_old_models(prefixes):
    old_models_dir = 'Models/old_models'
    if not os.path.exists(old_models_dir):
        os.makedirs(old_models_dir)
    # List of files to avoid moving (only filenames)
    files_to_avoid = ['tfidf_vectorizer_xgb_priority.pkl', 'tfidf_vectorizer_xgboost.pkl']
    for prefix in prefixes:
        for model_path in glob.glob(f'Models/{prefix}*.pkl'):
            filename = os.path.basename(model_path)
            if filename not in files_to_avoid:
                shutil.move(model_path, old_models_dir)
#Train Model
def train_model():
    issue_df = pd.read_csv("Data/Training/Categorization/train.csv",delimiter="\t")
    issue_df.drop_duplicates()
    issue_df.dropna()
    X  = issue_df['Error']
    y = issue_df['Sub_Issue']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
    X_test_tfidf = tfidf_vectorizer.transform(X_test)  
    svm_classifier = SVC(kernel='linear')  
    svm_classifier.fit(X_train_tfidf, y_train)
    y_pred = svm_classifier.predict(X_test_tfidf)
    joblib.dump(svm_classifier, 'Models/trained_model.pkl') 
    X  = issue_df['Error'].apply(pre_preocess_errors)
    y = issue_df['Sub_Issue']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
    X_test_tfidf = tfidf_vectorizer.transform(X_test) 
    base_models = [
        ('svc', svm_classifier),
    ]
    # Create a meta-learner (e.g., Logistic Regression)
    #meta_learner = RandomForestClassifier()
    meta_learner = RandomForestClassifier(n_estimators=300, random_state=42,criterion="entropy",
                                              max_depth= None, min_samples_leaf=1, min_samples_split= 2,max_features= 'log2')
    # Create a StackingClassifier
    stacking_classifier = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner
    )
    # Fit the stacking classifier on the training data
    stacking_classifier.fit(X_train_tfidf, y_train)
    # Make predictions on the test data
    y_pred = stacking_classifier.predict(X_test_tfidf)
    # Evaluate the stacking classifier
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(stacking_classifier,'Models/stacked_classifier.pkl')
    joblib.dump(tfidf_vectorizer, 'Models/tfidf_vectorizer.pkl')
    return str(accuracy)
#Function to check has errors or not
def has_more_than_one_line(file_path):
    try:
        with open(file_path, 'r') as file:
            num_lines = sum(1 for _ in file)
            return num_lines > 1
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
# In[20]:
def pre_preocess_errors(error_message):
    error_message = error_message.strip()
    # Define a regex pattern to match and replace other numbers (not common error codes)
    error_message = re.sub(r'(?<!\d)(?<!404)(?<!500)(?<!401)(?<!403)\d(?!04)(?!00)(?!01)(?!03)\d*(?!\d)', '', error_message)
    # Remove email addresses, URLs, and other characters
    error_message = re.sub(r'https?://\S+', '', error_message)
    error_message = re.sub(r'\S+@\S+', '', error_message)
    error_message = re.sub(r'[.,:\'"]', '', error_message)
    # Remove extra spaces and return the processed error message
    error_message = ' '.join(error_message.split())
    return error_message
from fuzzywuzzy import fuzz
def check_similarity(text_to_check, df):
    threshold = 98
    for index, row in df.iterrows():
        if isinstance(text_to_check, str) and isinstance(row['Error'], str):
            similarity_score = fuzz.ratio(text_to_check, row['Error'])
            if similarity_score >= threshold:
                return True, index
    return False, None
def get_unique_file_name():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data_{timestamp}"
    return filename
# In[21]:
def incremental_model_train_catogorization():
    # Move old models if they exist
    model_prefixes = ['trained_model', 'stacked_classifier', 'tfidf_vectorizer']
    move_old_models(model_prefixes)
    issue_df = pd.read_csv("Data/Training/Categorization/train.csv",delimiter="\t")
    issue_df.drop_duplicates()
    issue_df.dropna()
    X  = issue_df['Error']
    y = issue_df['Sub_Issue']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
    X_test_tfidf = tfidf_vectorizer.transform(X_test)  
    svm_classifier = SVC(kernel='linear')  
    svm_classifier.fit(X_train_tfidf, y_train)
    y_pred = svm_classifier.predict(X_test_tfidf)
    X  = issue_df['Error'].apply(pre_preocess_errors)
    y = issue_df['Sub_Issue']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
    X_test_tfidf = tfidf_vectorizer.transform(X_test) 
    base_models = [
        ('svc', svm_classifier),
    ]
    meta_learner = RandomForestClassifier(n_estimators=300, random_state=42,criterion="entropy",
                                              max_depth= None, min_samples_leaf=1, min_samples_split= 2,max_features= 'log2')
    stacking_classifier = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner
    )
    stacking_classifier.fit(X_train_tfidf, y_train)
    y_pred = stacking_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print("incremental learning acuracy is",accuracy)
    if(accuracy < model_threshold):
        return None
    # Get the current date and time
    now = datetime.now()
    # Format the current date and time
    formatted_now = now.strftime("%Y-%m-%d")
    joblib.dump(svm_classifier, f'Models/trained_model_{formatted_now}.pkl') 
    joblib.dump(stacking_classifier,f'Models/stacked_classifier_{formatted_now}.pkl')
    joblib.dump(tfidf_vectorizer, f'Models/tfidf_vectorizer_{formatted_now}.pkl')
    return str(accuracy)
# incremental_model_train_catogorization()
# In[22]:
def re_train_catogrization(path_to_new_df):
    new_df = pd.read_csv(path_to_new_df,delimiter="\t")
    old_df = pd.read_csv("Data/Training/Categorization/train.csv",delimiter="\t")
    for index, row in new_df.iterrows():
        new_error = row["Error"]
        new_issue = row["Issue"]
        new_sub_issue = row["Sub_Issue"]
        exsists, index = check_similarity(new_error,old_df)
        if (exsists):
            old_df.at[index, "Issue"] = new_issue
            old_df.at[index, "Sub_Issue"] = new_sub_issue
        else:
            new_row = {'Error': new_error, 'Issue': new_issue ,'Sub_Issue': new_sub_issue}
            new_row_df = pd.DataFrame([new_row])
            old_df = pd.concat([old_df, new_row_df], ignore_index=True)
    source_file = 'Data/Training/Categorization/train.csv'  
    destination_folder = 'Data/Training/Categorization/Old/'
    unique_file_name = get_unique_file_name() +".csv"
    destination_file = os.path.join(destination_folder, unique_file_name)
    shutil.move(source_file, destination_file)
    old_df.to_csv("Data/Training/Categorization/train.csv",sep = "\t",index=False)
def retrain_resolution_model(path_to_new_df):
    new_df = pd.read_csv(path_to_new_df,delimiter="\t")
    old_df = pd.read_csv('Data/Training/Resolution/resolution.csv')
    for index, row in new_df.iterrows():
        new_error = row["Error"]
        resolution = row["Resolution"]
        exsists, index = check_similarity(new_error,old_df)
        if (exsists):
            old_df.at[index, "Resolution"] = resolution
        else:
            new_row = {'Error': new_error, 'Resolution': resolution}
            new_row_df = pd.DataFrame([new_row])
            old_df = pd.concat([old_df, new_row_df], ignore_index=True)
    source_file = 'Data/Training/Resolution/resolution.csv'  
    destination_folder = 'Data/Training/Resolution/Old/'
    unique_file_name = get_unique_file_name() +".csv"
    destination_file = os.path.join(destination_folder, unique_file_name)
    shutil.move(source_file, destination_file)
    old_df.to_csv("Data/Training/Resolution/resolution.csv",index=False)
def re_train_models():
    # add the Csv here
    new_data = 'New_data/new_data.csv'
    if os.path.exists(new_data):
        re_train_catogrization(new_data)
        try:
            incremental_model_train_catogorization()
        except shutil.Error as e:
            print(f"Error moving model ': {e}")
        print("Strat retrain_resolution_model")
        retrain_resolution_model(new_data)
        print("end retrain_resolution_model")
        incremental_model_train_resolution()
        # remove the csv 
        move_csv_file()
def start_retrain():
    thread = threading.Thread(target=re_train_models)
    thread.start()
def schedule_job():
    schedule.every().day.at("00:01").do(start_retrain)
    while True:
        schedule.run_pending()
        time.sleep(1)
job_thread = threading.Thread(target=schedule_job)
job_thread.start()
# In[26]:
def move_csv_file():
    current_date = datetime.now().strftime("%Y-%m-%d")
    source_path = "New_data/new_data.csv"
    destination_path = f"New_data/Old_data/old_data_{current_date}.csv"
    if os.path.exists(source_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(source_path, destination_path)
        print(f"File moved and renamed to old_data_{current_date}.csv")
    else:
        print("Source file does not exist.")
from datetime import datetime
def get_month_file_suffix(custom_date):
    # Extract month and year from the custom date
    month = datetime.strptime(custom_date, "%m/%d/%Y").strftime("%B").lower()
    year = datetime.strptime(custom_date, "%m/%d/%Y").strftime("%Y")
    # Construct the file suffix based on month and year
    file_suffix = f"{month}_{year}.csv"
    return file_suffix
def convert_int64_to_python_types(obj):
    if isinstance(obj, pd.DataFrame):
        # Convert DataFrame columns to_dict
        for col in obj.columns:
            obj[col] = obj[col].to_list()
    elif isinstance(obj, pd.Series):
        # Convert Series to list
        obj = obj.to_list()
    elif isinstance(obj, np.int64):
        # Convert np.int64 to int
        obj = int(obj)
    elif isinstance(obj, dict):
        # Recursively convert values in dictionary
        for key, value in obj.items():
            obj[key] = convert_int64_to_python_types(value)
    elif isinstance(obj, list):
        # Recursively convert values in list
        obj = [convert_int64_to_python_types(item) for item in obj]
    return obj
def assign_work(work, time_to_finish, time_format="%m/%d/%Y", max_work_hours=9, days_skip=0):
    work_date = datetime.now().strftime(time_format)
    file_suffix = get_month_file_suffix(work_date)
    file_name = "Assign_work/"f"employee_data_wee_{file_suffix}"
    work_flow_df = pd.read_csv(file_name)
    index_of_work_date = work_flow_df.columns.get_loc(work_date)
    next_column_index = index_of_work_date + days_skip
    if next_column_index < len(work_flow_df.columns):
        work_date = work_flow_df.columns[next_column_index]
        selected_data = work_flow_df[['Employee', work_date]]
    else:
        # No available time
        return {
            "success": False,
            "message": "No available time",
            "assigned_employee": None
        }
    min_data = selected_data.nsmallest(1, work_date)
    selected_employee_data = None
    if min_data.duplicated(subset=[work_date]).any():
        selected_employee_data = min_data.sample(n=1)
    else:
        selected_employee_data = min_data
    selected_employee_index = selected_employee_data.index[0]
    selected_employee = selected_employee_data['Employee'].iloc[0]
    selected_employee_email = work_flow_df.loc[selected_employee_index, 'Email']
    selected_hours = selected_employee_data[work_date].iloc[0]
    new_employee_hours = selected_hours + time_to_finish
    if new_employee_hours > max_work_hours:
        # Implement the priority
        return assign_work(
            work,
            time_to_finish=time_to_finish,
            days_skip=days_skip + 1,
            time_format=time_format,
            max_work_hours=max_work_hours
        )
    work_flow_df.at[selected_employee_index, work_date] = new_employee_hours
    work_flow_df.to_csv("Assign_work/employee_data_weekdays_november.csv", index=False)
    # Return the details in JSON format
    result = {
        "success": True,
        # "message": f"Work Assigned to {selected_employee}. New employee hours are {new_employee_hours}. Selected date is {work_date}.",
        "assigned_employee": {
            "name": selected_employee,
            "email": selected_employee_email,
            # "hours": new_employee_hours,
            "date": work_date
        }
    }
    return result
# In[28]:
def predict_priority(error_message):
    # Preprocess the input error message
    error_message_tfidf = tfidf_vectorizer_priority.transform([error_message])
    # Make the prediction
    predicted_label_encoded = xgb_classifier_priority.predict(error_message_tfidf)
    # Decode the predicted label
    predicted_priority = label_encoder_priority.inverse_transform(predicted_label_encoded)[0]
    return predicted_priority
def filter_technical_terms(summary):
    prompt_message=f"""
    Instructions: Answer directly without any preamble or explanations.
    you are an expert of text summerization with decades of expereance in summrizing texts that are
    tecnical that can be understandable to people with out any tecnicaly expereance
    your job is to analize the given text and summerize the content in a way that is short and a
    non tecnical persion can understand the full picture with out the knloage of any tecnical terms 
    still give the full picture of the text 
    here is the text to analize
    {summary}    
    """
    llm = VertexAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    max_retries=5,  
    retry_on_timeout=True,  
    )
    output = llm.invoke(prompt_message.format(summary=summary))
    return [output.replace('*','')]
# Function to summarize an error and filter technical terms
def summarize(error):
    # Generate summary
    # Filter out technical terms from the summary
    non_technical_summary = filter_technical_terms(error)
    return non_technical_summary
# In[30]:
def extract_name_and_convert_date(assignee_details):
    assignee_name = assignee_details["assigned_employee"]["email"]
    assignee_date_str = assignee_details["assigned_employee"]["date"]
    return assignee_name, convert_date_format(assignee_date_str)
def convert_date_format(date_str, input_format="%m/%d/%Y", output_format="%Y-%m-%d"):
    try:
        date_obj = datetime.strptime(date_str, input_format)
        return f'{date_obj.strftime(output_format)}'
    except ValueError:
        return None
def train_priority_model():
    # Load data
    issue_df = pd.read_csv("Data/Training/Priority/output.csv")
    issue_df = issue_df.dropna()
    # Split data
    X = issue_df['Error']
    y = issue_df['Priority']
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(X)
    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    # Create and train XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    # Make predictions
    preds = xgb_model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:", accuracy)
    # Save models
    joblib.dump(tfidf_vectorizer, 'Models/tfidf_vectorizer_xgb_priority.pkl')
    joblib.dump(xgb_model, 'Models/XgBoost_Classifier_priority.pkl')
    joblib.dump(label_encoder, 'Models/label_encoder_xgb_priority.pkl')
import requests
app = Flask(__name__)
CORS(app)
#config for uploads and upload files ext
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'log', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#check to see is a allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def list_routes():
    """
    List all routes, HTTP methods, and docstrings in the Flask app.
    Returns:
    - List of dictionaries, each containing route, methods, and docstring.
    """
    output = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            route = {
                "route": str(rule),
                "methods": [method for method in rule.methods if method != 'HEAD' and method != 'OPTIONS'],
                "docstring": app.view_functions[rule.endpoint].__doc__ if app.view_functions.get(rule.endpoint) else ""
            }
            output.append(route)
    return output
@app.route('/add-new-data', methods=['POST'])
def process_json():
    json_data = request.get_json()
    try:
        error = json_data['error']
        issue = json_data['issue']
        sub_issue = json_data['sub_issue']
        resolution = json_data['resolution']
        folder_name = 'New_data'
        file_path = f"{folder_name}/new_data.csv"
        if not os.path.isfile(file_path):
            data = {
                'UUID': [],
                'Error': [],
                'Issue': [],
                'Sub_Issue': [],
                'Resolution': []
            }
            df = pd.DataFrame(data)
            df.to_csv(file_path, sep='\t', index=False)
        else:
            df = pd.read_csv(file_path, delimiter="\t")
        # Duplicate check
        if error in df['Error'].values:
            existing_details = df[df['Error'] == error].to_dict(orient='records')[0]
            response = {
                "message": "Error already exists",
                "details": {
                    "uuid": existing_details['UUID'],
                    "error": existing_details['Error'],
                    "issue": existing_details['Issue'],
                    "sub_issue": existing_details['Sub_Issue'],
                    "resolution": existing_details['Resolution']
                }
            }
            return jsonify(response)
        else:
            # Trying to insert a new record
            return jsonify({
                "message": "Trying to insert a new record"
            })
    except KeyError:
        return jsonify({"error": "Missing keys in JSON data"}), 400
@app.route('/update-data', methods=['PUT'])
def update_record():
    json_data = request.get_json()
    try:
        uuid_to_update = json_data.get('uuid')
        issue = json_data.get('issue')
        sub_issue = json_data.get('sub_issue')
        resolution = json_data.get('resolution')
        folder_name = 'New_data'
        file_path = f"{folder_name}/new_data.csv"
        if not os.path.isfile(file_path):
            return jsonify({"error": "File not found"}), 404
        df = pd.read_csv(file_path, delimiter="\t")
        # Check if the UUID exists in the DataFrame
        if pd.isna(uuid_to_update) or uuid_to_update not in df['UUID'].values:
            # If UUID is empty or not found, treat as a new record
            error = json_data.get('error')
            new_uuid = str(uuid.uuid4())
            new_data = {
                'UUID': new_uuid,
                'Error':error,
                'Issue': issue,
                'Sub_Issue': sub_issue,
                'Resolution': resolution
            }
            new_entry = pd.DataFrame([new_data])
            # Add new Record
            frames = [df, new_entry]
            df = pd.concat(frames, ignore_index=True)
            df.to_csv(file_path, sep='\t', index=False)
            return jsonify({
                "message": "New record added to CSV file successfully",
            })
        else:
            # Update existing record
            df.loc[df['UUID'] == uuid_to_update, ['Issue', 'Sub_Issue', 'Resolution']] = issue, sub_issue, resolution
            df.to_csv(file_path, sep='\t', index=False)
            return jsonify({"message": "Record updated successfully"})
    except KeyError:
        return jsonify({"error": "Missing keys in JSON data"}), 400
# @app.route('/assign-work', methods=['GET'])
def assign_work_endpoint():
    try:
        data = request.json
        # Set default values if not provided in the request
        work = data.get('work', 'Default Work')
        time_to_finish = data.get('time_to_finish', 1)
        time_format = data.get('time_format', "%m/%d/%Y")
        max_work_hours = data.get('max_work_hours', 9)
        days_skip = data.get('days_skip', 0)
        result = assign_work(work, time_to_finish, time_format, max_work_hours, days_skip)
        # Convert int64 types to standard Python types
        result = convert_int64_to_python_types(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"An error occurred: {str(e)}",
            "assigned_employee": None
        })
@app.route('/jira-create', methods=['POST'])
def create_issue():
    if request.method == 'POST':
        data = request.get_json()
        project = data.get('project')
        summary = data.get('summary')
        description = data.get('description')
        username=data.get('username')
        password=data.get('password')
        priority=data.get('priority')
        url=data.get('url')
        if priority == "high":
            priority = "High"
        elif priority == "medium":
            priority = "Medium"
        elif priority == "low":
            priority = "Low"
        else:
            return jsonify({"error": "Invalid priority value"}), 400
        if not url:
            url = "https://hive-accello-dev.atlassian.net"
        #Get The Responce
        issue_id = create_defect(project, summary, description,username,password,priority,url)
        if issue_id:
            return jsonify({"ticketId": issue_id})
        else:
            return "Failed to create the issue", 500
def create_defect(project_name, summary, description,username,password,priority,url):
    create_issue_url = f"{url}/rest/api/3/issue/"
    headers = {'Content-Type': 'application/json'}
    auth = (username, password)
    assignee_details = assign_work(None, time_to_finish=2)
    assignee_name, start_date = extract_name_and_convert_date(assignee_details)
    print(start_date)
    print(assignee_name)
    description_field = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": description
                    }
                ]
            }
        ]
    }
    fields = {
        "fields": {
            "project": {"key": project_name},
            "summary": summary,
            "description": description_field,
            "assignee":{"id":assignee_name},
            "customfield_10015": start_date,  
            "priority": {"name": priority},  
            "issuetype": {"name": "Bug"},
            "customfield_10060":{
                "id": "10079",
            },
            "customfield_10059":"19",
        }
    }
    response = requests.post(create_issue_url, headers=headers, auth=auth, data=json.dumps(fields))
    if response.status_code // 100 == 2: 
        try:
            issue_key = response.json()["key"]
            return issue_key
        except KeyError as e:
            print(f"Unable to extract issue key: {e}")
            return None
    else:
        print(f"Error creating issue: {response.text}")
        return None
#Endpoint for get predictions from xgboost Model
@app.route('/get-prediction',methods=['POST'])
def get_predictions_xgboost():
    """Get prediction.
    Request Format:
    - Method: GET
    - Query Parameter:
      - error (str): The error message for prediction.
    Example:
    /get-prediction?error=example_error
    Returns:
    - JSON response with prediction information.
    """
    data = request.get_json()
    # Access specific keys in the JSON data
    value = data.get('error',None)
    #input_text=data['text']
    #if input_txt 
    if value is None:
        return jsonify({'message':'Error Not defined'})
    predicion, probability_percentages = get_prediction_stacked_value(value)
    summary=summarize(value)
    priority=predict_priority(value)
    if predicion != "Load the model and the encoder":
        issue = get_issue(predicion)
        resolution_input_text = value
        start_time = time.time()
        resolution =  get_resolution(resolution_input_text)
        end_time=time.time()
        resolution_time=time_to_execte = start_time - end_time
        print(f"Prediction time: {prediction_time} seconds, Summary time: {summary_time} seconds, Resolution time: {resolution_time} seconds")
        json_data = json.dumps({'sub_issue': predicion,
                           'issue':issue,
                           'resolution':resolution,
                           'error':value,
                           'percentage':probability_percentages,
                           'summary':summary,
                           'priority':priority,
                               })
        return json_data
    return json.dumps({'message': "Load the model and the encoder"})
#Make List Of Predictions 
@app.route("/get-predictions",methods=['POST'])
def get_predictions_list():
    """Get predictions for a list of errors using XGBoost.
    Request Format:
    - Method: POST
    - JSON Body:
      - errors (list of str): A list of error messages for prediction.
    Example JSON Request:
    {
        "errors": ["error1", "error2", "error3"]
    }
    Returns:
    - JSON response with prediction information.
    - If successful, a JSON response containing a list of predictions.
    - If the model and encoder are not loaded, an error response is returned.
    Example JSON Response (Successful):
    [
        {"error": "error1", "prediction": "prediction1"},
        {"error": "error2", "prediction": "prediction2"},
        {"error": "error3", "prediction": "prediction3"}
    ]
    Example JSON Response (Error - Model Not Loaded):
    {"message": "Load the model and the encoder"}
    """
    data = request.get_json()
    # Access specific keys in the JSON data
    value = data.get('errors')
    if value is None:
        return jsonify({'message':'Error Not defined'})
    df = pd.DataFrame(value,columns=["Error"])
    predicions = get_predections_csv_stacked_list(df)
    if isinstance(predicions, pd.DataFrame):
        json_string = predicions.to_json(orient='records')
        return json_string
    return jsonify({'message': 'Load the model and the encoder'})
def delete_file(file_path):
    try:
        # Attempt to delete the file
        os.remove(file_path)
        return f"File '{file_path}' deleted successfully."
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except PermissionError:
        return f"Error: Permission denied to delete '{file_path}'."
    except Exception as e:
        return f"Error: {str(e)}"
def get_predections_csv_stacked(csv_file):
    global stacked_model
    global tfid_vec 
    if stacked_model is not None and tfid_vec is not None:
        #Loading The CSV File to test
        df = pd.read_csv(csv_file)
        #Getting the Feature and Target Values 
        X_test_data = df['Error'].apply(pre_preocess_errors)
        encoded_test_X = tfid_vec.transform(X_test_data)  
        #Getting Predections
        test_predictions = stacked_model.predict(encoded_test_X) 
        # Inverse Transform to Decode Predictions
        new_probability_estimates = stacked_model.predict_proba(encoded_test_X)
        results_df = pd.DataFrame({
        'error': df['Error'],
        'sub_issue' : test_predictions,
        'issue' :None,
        'resolution':None,
        'percentage':round(np.max(new_probability_estimates) * 100, 2),
        'summary': None,
        'priority':None     
        })
        summaries = []
        for error_text in X_test_data:
            summary = summarize(error_text)
            summaries.append(summary)
        # Add the summaries to the results_df
        results_df['summary'] = summaries
        for index, row in results_df.iterrows():
            results_df.at[index, 'issue'] = get_issue(row['sub_issue'])
            results_df.at[index, 'resolution'] = resolution_llm_gemeni(row['error'])
            results_df.at[index, 'summary'] = summarize(row['error'])
            results_df.at[index, 'priority'] = predict_priority(row['error'])
        return results_df
    else:
        return ("Load the model and the encoder")
#Get Predictions from a log xgboost
@app.route('/get-log-predictions', methods=['POST'])
def get_predictions_csv_parser_xgboost():
    """Get predictions from a log file.
    Request Format:
    - Method: POST
    - JSON Body:
      - file_name (str): The name of the log file to process.
    Example JSON Request:
    {
        "file_name": "example.log"
    }
    Returns:
    - JSON response with prediction information.
    - If successful, a JSON response containing predictions from the log file.
    - If the file is invalid or does not contain data, an error response is returned.
    - If the model and encoder are not loaded, an error response is returned.
    """
    data = request.get_json()
    value = data.get('file_name',None)
    if value is None:
        return jsonify({'message':'File name not defined'})
    if not os.path.isfile(value):
         return jsonify({'message': 'Invalid File'})
    file_name = process_log(value)
    if(has_more_than_one_line(file_name)):
        predicions = get_predections_csv_stacked(file_name)
        if isinstance(predicions, pd.DataFrame):
            json_string = predicions.to_json(orient='records')
            print("file name ", file_name)
            delete_file(file_name)
            delete_file(value)
            return json_string
        delete_file(file_name)
        delete_file(value)
        return jsonify({'message': 'Load the model and the encoder'})
    else:
        delete_file(file_name)
        delete_file(value)
        return jsonify({'message': 'No Data'})
@app.route('/get-models', methods=['GET'])
def get_models():
    models_directory = "Models/old_models"
    models_info = []
    for filename in os.listdir(models_directory):
        if filename.startswith("stacked_classifier") and filename.endswith(".pkl"):
            # Extract the date from the filename
            try:
                date_part = filename.split('_')[-1].split('.')[0]
                publish_date = datetime.strptime(date_part, "%Y-%m-%d").date()
            except ValueError:
                # If the filename doesn't match the expected format, skip it
                continue
            model_info = {
                "Name": f"model_TFA_{publish_date.strftime('%Y_%m_%d')}",
                "Author": "Admin",
                "Publish_DateTime": publish_date.strftime("%Y-%m-%d 00:00"),
                "Trained_DateTime": publish_date.strftime("%Y-%m-%d 00:00")
            }
            models_info.append(model_info)
    # models_info.sort(key=lambda x: datetime.strptime(x["Name"].split('_')[-1], "%Y_%m_%d"), reverse=True)
    reversed_models_info = list(reversed(models_info))
    return jsonify(reversed_models_info)
@app.route('/get-log-predictions-multiple-files', methods=['POST'])
def get_predictions_multiple_files_csv_parser_xgboost():
    data = request.get_json()
    file_names = data.get('file_names', [])
    model_name = data.get('model', None)
    if not file_names:
        return jsonify({'message': 'File names not defined'})
    global stacked_model
    global tfid_vec
    current_model = stacked_model
    current_vectorizer = tfid_vec
    if model_name:
        match = re.search(r'_([0-9]{4}_[0-9]{2}_[0-9]{2})$', model_name)
        if match:
            model_date = match.group(1).replace('_', '-')
            loaded_model, loaded_vectorizer = load_model_and_vectorizer(model_date)
            if loaded_model is None or loaded_vectorizer is None:
                return jsonify({'message': 'Model files not found'})
            current_model = loaded_model
            current_vectorizer = loaded_vectorizer
        else:
            return jsonify({'message': 'Invalid model name format'})
    results = {}
    for file_name in file_names:
        if not os.path.isfile(file_name):
            results[file_name] = {'message': 'Invalid File'}
        else:
            file_path = process_log(file_name)
            if has_more_than_one_line(file_path):
                predictions = get_predections_csv_stacked(file_path, current_model, current_vectorizer)
                if isinstance(predictions, pd.DataFrame):
                    results[file_name] = predictions.to_dict(orient='records')
                else:
                    results[file_name] = {'message': 'Load the model and the encoder'}
            else:
                results[file_name] = {'message': 'No Data'}
        delete_file(file_path)
        delete_file(file_name)
    return jsonify(results)
#Endpoint for uploading a log file to server
@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload a log file to the server.
    Request Format:
    - Method: POST
    - Form Data:
      - file (file): The log file to upload.
    Returns:
    - JSON response with the uploaded file path.
    - If the file is successfully uploaded, a JSON response containing the file path is returned.
    - If no file is provided, an error response is returned.
    - If the selected file has an invalid format, an error response is returned.
    Example JSON Response (Successful):
    {"file_path": "/uploads/example.log"}
    Example JSON Response (Error - No File Part):
    {"message": "No file part"}
    Example JSON Response (Error - No Selected File):
    {"message": "No selected file"}
    Example JSON Response (Error - Invalid File Format):
    {"message": "Invalid file format"}
    """
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'})
    directory = app.config['UPLOAD_FOLDER']
    if not os.path.exists(directory):
        os.makedirs(directory)
    if file and allowed_file(file.filename):
        if ".csv" in file.filename:
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER+"/New_Data"
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        return jsonify({'file_path': file_path})
    return jsonify({'message': 'Invalid file format'})
# Global DataFrame to store the uploaded file
global_df = None
def is_non_string(value):
    # Check if the value can be converted to a float (numeric)
    try:
        float(value)
        return True
    except ValueError:
        pass
    # Check if the value matches time formats (e.g., "12:12:16 AM" or "12:03.2")
    time_pattern_1 = re.compile(r'^\d{1,2}:\d{2}:\d{2}\s?[APap][Mm]$')  # Matches "12:12:16 AM"
    time_pattern_2 = re.compile(r'^\d{1,2}:\d{2}\.\d$')  # Matches "12:03.2"
    if isinstance(value, str) and (time_pattern_1.match(value) or time_pattern_2.match(value)):
        return True
    return False
###########################################################################################
#modified with the .csv file extraction for existing endpoint
@app.route('/upload_files', methods=['POST'])
def upload_files():
    global global_df
    """Upload log files to the server."""
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})
    files = request.files.getlist('file')  # Retrieve a list of files from the request
    if not files or all(file.filename == '' for file in files):
        return jsonify({'message': 'No selected files'})
    uploaded_file_paths = []
    string_column_names = []
    # Process each file in the list
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Save and process files based on their type
            if filename.endswith(".csv"):
                try:
                    # Read the CSV file
                    global_df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))
                    # Filter columns that contain only string values
                    string_column_names = [
                        col for col in global_df.columns
                        if global_df[col].apply(lambda x: not is_non_string(x) and isinstance(x, str)).all()
                    ]
                    # Save the CSV file to the New_Data directory
                    new_data_directory = os.path.join(app.config['UPLOAD_FOLDER'], "New_Data")
                    if not os.path.exists(new_data_directory):
                        os.makedirs(new_data_directory)
                    file_path = os.path.join(new_data_directory, filename)
                    file.save(file_path)
                except Exception as e:
                    return jsonify({'message': f'Error processing CSV file: {str(e)}'}), 500
            else:
                try:
                    # Save non-CSV files to the default upload folder
                    file.save(file_path)
                    uploaded_file_paths.append(file_path)
                except Exception as e:
                    return jsonify({'message': f'Error saving file: {str(e)}'}), 500
    if string_column_names:
        return jsonify({'columns': string_column_names}), 200
    return jsonify({'file_paths': uploaded_file_paths})
#Endpoint for get extracted erros from a log
@app.route('/extract-errors',methods=['POST'])
def get_extracted_errors():
    """Extract errors from a log file and return as a downloadable file.
    Request Format:
    - Method: POST
    - JSON Body:
      - file_name (str): The name of the log file to process.
    Example JSON Request:
    {
        "file_name": "example.log"
    }
    Returns:
    - Downloadable file response containing the extracted errors.
    - If successful, the log file containing extracted errors is returned as a downloadable file.
    - If the file is invalid or does not exist, an error response is returned.
    Example Response (Successful):
    - A downloadable file containing the extracted errors from the log file.
    Example JSON Response (Error - File Name Not Defined):
    {"message": "File name Not defined"}
    Example JSON Response (Error - Invalid File):
    {"message": "Invalid File"}
    """
    data = request.get_json()
    value = data.get('file_name')
    if value is None:
        return jsonify({'message':'File name Not defined'})
    if not os.path.isfile(value):
         return jsonify({'message': 'Invalid File'})
    file_name = process_log(value)
    try:
        return send_file(file_name)
    except Exception as e:
        return jsonify({'message': str(e)})
#Endpoint for uploading csv file to the dataset
# @app.route('/update-model',methods=['POST'])
def update_model():
    """Update the training model with new data.
    Request Format:
    - Method: POST
    - JSON Body:
      - file_name (str): The name of the CSV file containing new data.
      - delemeater (str, optional): The delimiter used in the CSV file (default: ",").
    Example JSON Request:
    {
        "file_name": "new_data.csv",
        "delemeater": ","
    }
    Returns:
    - JSON response with update status.
    - If successful, the training model is updated with new data, and an update status is returned.
    - If the file is invalid or does not exist, an error response is returned.
    - The optional delimiter parameter specifies the delimiter used in the CSV file.
    Example JSON Response (Successful):
    {"message": "Model updated successfully"}
    Example JSON Response (Error - File Name Not Defined):
    {"message": "File name Not defined"}
    Example JSON Response (Error - Invalid File):
    {"message": "Invalid File"}
    """
    data = request.get_json()
    value = data.get('file_name')
    if value is None:
        return jsonify({'message':'File name Not defined'})
    delemeater = data.get('delemater',",")
    if not os.path.isfile(value):
         return jsonify({'message': 'Invalid File'})
    original_df = pd.read_csv("Data/Training/Categorization/train.csv", delimiter="\t")
    new_df = pd.read_csv(value,delimiter=delemeater)
    local_timezone = tz.tzlocal()
    current_time = datetime.datetime.now(local_timezone)
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S_%Z")
    old_file_path = "Data/Training/Categorization/Old/"+formatted_time+".csv"
    original_df.to_csv(old_file_path,index=False,sep="\t")
    merged_df = pd.concat([original_df, new_df], ignore_index=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.to_csv("Data/Training/Categorization/train.csv",index=False,sep="\t")
    return train_model()
@app.route("/get-old-models",methods=['GET'])
def get_older_models():
    """Get a list of older model directories.
    Request Format:
    - Method: GET
    Returns:
    - JSON response with a list of older model directories.
    - The response includes directories for both XGBoost and CNN models.
    - Each model directory represents an older version of the model.
    Example JSON Response:
    {
        "XGBoost": ["model1", "model2"],
        "CNN": ["modelA", "modelB"]
    }
    """
    xgboost_directory_path = "Models/Old_XGboost"
    xgboost_all_items = os.listdir(xgboost_directory_path)
    xgboost_directories = [item for item in xgboost_all_items if os.path.isdir(os.path.join(xgboost_directory_path, item))]
    cnn_directory_path = "Models/Old_Cnn"
    cnn_all_items = os.listdir(cnn_directory_path)
    cnn_directories = [item for item in cnn_all_items if os.path.isdir(os.path.join(cnn_directory_path, item))]
    json_string = {"XGBoost" : xgboost_directories,"CNN":cnn_directories}
    json_data = json.dumps(json_string, indent=4)
    return json_data
@app.route("/update-resolution-dataset",methods=['POST'])
def update_resolution_model():
    """Update the resolution model with new data.
    Request Format:
    - Method: POST
    - JSON Body:
      - file_name (str): The name of the CSV file containing new data.
      - delimiter (str, optional): The delimiter used in the CSV file (default: ",").
    Example JSON Request:
    {
        "file_name": "new_resolution_data.csv",
        "delimiter": ","
    }
    Returns:
    - JSON response with update status.
    - If successful, the resolution model is updated with new data, and an update status is returned.
    - If the file is invalid or does not exist, an error response is returned.
    - The optional delimiter parameter specifies the delimiter used in the CSV file.
    Example JSON Response (Successful):
    {"message": "Resolution model updated successfully"}
    Example JSON Response (Error - File Name Not Defined):
    {"message": "File name Not defined"}
    Example JSON Response (Error - Invalid File):
    {"message": "Invalid File"}
    """
    data = request.get_json()
    value = data.get('file_name')
    if value is None:
        return jsonify({'message':'File name Not defined'})
    delemeater = data.get('delemater',",")
    if not os.path.isfile(value):
         return jsonify({'message': 'Invalid File'})
    original_df = pd.read_csv("Data/Training/Resolution/resolution.csv", delimiter=",")
    new_df = pd.read_csv(value,delimiter=delemeater)
    local_timezone = tz.tzlocal()
    current_time = datetime.datetime.now(local_timezone)
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S_%Z")
    old_file_path = "Data/Training/Resolution/Old/"+formatted_time+".csv"
    original_df.to_csv(old_file_path,index=False,sep=",")
    merged_df = pd.concat([original_df, new_df], ignore_index=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.to_csv("Data/Training/Resolution/resolution.csv",index=False,sep=",")
    return trian_resolution_model()
@app.route("/load-categorie-old-model")
def load_old_model_cat():
    """Load an older categorization model.
    Request Format:
    - Method: GET
    - URL Parameter:
      - version (str): The version of the older model to load.
    Example URL:
    /load-categorie-old-model?version=model1
    Returns:
    - JSON response with the status of loading the older model.
    - If successful, the specified older categorization model is loaded, and a success message is returned.
    - If the version is not defined or the model does not exist, an error response is returned.
    Example JSON Response (Successful):
    {"message": "Models/Old_XGboost/model1 Model Loaded"}
    Example JSON Response (Error - Version Not Defined):
    {"message": "Version Not defined"}
    Example JSON Response (Error - Invalid Model):
    {"message": "Invalid value"}
    """
    data = request.get_json()
    value = data.get('version',None)
    if value is None:
        return jsonify({'message':'Version Not defined'})
    value = "Models/Old_XGboost/"+value
    if not os.path.isdir(value):
         return jsonify({'message': 'Invalid value'})
    global xgboost_model
    global tfidvec_xgboost_model
    global label_encoder_model
    xgboost_model= joblib.load(value+'/xgboost.pkl')
    tfidvec_xgboost_model = joblib.load(value+'/tfidf_vectorizer_xgboost.pkl')
    label_encoder_model =joblib.load(value+'/label_encoder.pkl')
    return jsonify({'message': value+' Model Loaded'})
@app.route("/reset-categorie-models")
def reset_old_model_cat():
    """Reset the categorization models to the default models.
    Request Format:
    - Method: GET
    Returns:
    - JSON response with the status of resetting the categorization models.
    - If successful, the categorization models are reset to the default models, and a success message is returned.
    Example JSON Response (Successful):
    {"message": "All categorization models reset"}
    """
    global xgboost_model
    global tfidvec_xgboost_model
    global label_encoder_model  
    xgboost_model = joblib.load('Models/xgboost.pkl')
    tfidvec_xgboost_model = joblib.load('Models/tfidf_vectorizer_xgboost.pkl')
    label_encoder_model = joblib.load('Models/label_encoder.pkl')
    return jsonify({'message': 'All catogorization models reseted'})
# @app.route("/load-resolution-old-model")
def load_old_resolution_model():
    """Load an older resolution model.
    Request Format:
    - Method: GET
    - JSON Data:
      - version (str): The version of the older model to load.
    Example JSON Data:
    {"version": "model1"}
    Returns:
    - JSON response with the status of loading the older model.
    - If successful, the specified older resolution model is loaded, and a success message is returned.
    - If the version is not defined or the model does not exist, an error response is returned.
    Example JSON Response (Successful):
    {"message": "Models/Old_Cnn/model1 Model Loaded"}
    Example JSON Response (Error - Version Not Defined):
    {"message": "Version Not defined"}
    Example JSON Response (Error - Invalid Model):
    {"message": "Invalid value"}
    """
    data = request.get_json()
    value = data.get('version')
    if value is None:
        return jsonify({'message':'Version Not defined'})
    value = "Models/Old_Cnn/"+value
    if not os.path.isdir(value):
         return jsonify({'message': 'Invalid value'})
    global cnn_model
    global cnn_label_encoder_model
    cnn_model = joblib.load(value+'/cnn_model.pkl')
    cnn_label_encoder_model = joblib.load(value+'/cnn_label_encoder.pkl')
    return jsonify({'message': value+' Model Loaded'})
# @app.route("/reset-resolution-model")
def reset_resolution_model_cat():
    """Reset the resolution model to the default model.
    Request Format:
    - Method: GET
    Returns:
    - JSON response with the status of resetting the resolution model.
    - If successful, the resolution model is reset to the default model, and a success message is returned.
    Example JSON Response (Successful):
    {"message": "All resolution models reseted"}
    """
    global cnn_model
    global cnn_label_encoder_model 
    cnn_model = joblib.load('Models/cnn_model.pkl')
    cnn_label_encoder_model = joblib.load('Models/cnn_label_encoder.pkl')
    return jsonify({'message': 'All catogorization models reseted'})
@app.route("/get-categorie-data")
def get_cat_data():
    """Get the training data for the categorization model.
    Request Format:
    - Method: GET
    Returns:
    - JSON response with the training data for the categorization model.
    Example JSON Response:
    [{"Error": "error_message_1"}, {"Error": "error_message_2"}, ...]
    """
    df = pd.read_csv("Data/Training/Categorization/train.csv",delimiter="\t")
    json_data = df.to_json(orient='records')
    return jsonify(json_data)
@app.route("/get-resolution-data")
def get_resolution_data():
    """Get the training data for the resolution model.
    Request Format:
    - Method: GET
    Returns:
    - JSON response with the training data for the resolution model.
    Example JSON Response:
    [{"Error": "error_message_1", "Resolution": "resolution_1"}, {"Error": "error_message_2", "Resolution": "resolution_2"}, ...]
    """
    df = pd.read_csv("Data/Training/Resolution/resolution.csv")
    json_data = df.to_json(orient='records')
    return jsonify(json_data)
@app.route('/s3_config', methods=['POST'])
def configure_s3():
    tenant = request.headers.get('X-TenantId')
    if not tenant:
        return jsonify({'error': 'Headder Value required'}), 400
    g.tenant = tenant
    # Get the configuration from the request data
    aws_access_key_id = request.json.get('aws_access_key_id')
    aws_secret_access_key = request.json.get('aws_secret_access_key')
    s3_bucket_name = request.json.get('s3_bucket_name')
    region_name = request.json.get('region_name')
    project_id = request.json.get('project_id')
    if not aws_access_key_id or not aws_secret_access_key or not s3_bucket_name or not region_name or not project_id:
        return jsonify({'error': 'All fields are required'}), 400
    try:
        upsert_s3_config(aws_access_key_id, aws_secret_access_key, s3_bucket_name, region_name, project_id)
        return jsonify(
        {
            'status':True,
            'message': 'S3 configuration updated successfully'
        }), 200
    except Exception as e:
        return jsonify(
            {
                'status':False,
                'message': str(e)}), 500
@app.route('/list_files', methods=['GET'])
def list_files_in_s3():
    tenant = request.headers.get('X-TenantId')
    if not tenant:
        return jsonify({'error': 'Headder Value required'}), 400
    project_id = request.json.get('project_id')
    if not project_id:
        return jsonify({'error': 'project id fields are required'}), 400
    g.tenant = tenant
    s3_client,s3_bucket_name = get_s3_connection(project_id)
    if not s3_client or not s3_bucket_name:
        return jsonify({'error': 'S3 is not configured'}), 400
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name)
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            return jsonify({'files': files}), 200
        else:
            return jsonify({'files': []}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/upload_folder', methods=['POST'])
def upload_fixed_folder_to_s3():
    tenant = request.headers.get('X-TenantId')
    if not tenant:
        return jsonify({'error': 'Headder Value required'}), 400
    project_id = request.json.get('project_id')
    if not project_id:
        return jsonify({'error': 'project id fields are required'}), 400
    g.tenant = tenant
    s3_client,s3_bucket_name = get_s3_connection(project_id)
    # Check if S3 is configured
    if not s3_client or not s3_bucket_name:
        return jsonify({'error': 'S3 is not configured'}), 400
    folder_path = "Logs"
    if not os.path.isdir(folder_path):
        return jsonify({'error': f'{folder_path} is not a valid directory'}), 400
    try:
        # Iterate through all files in the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = os.path.relpath(file_path, folder_path)
                s3_client.upload_file(file_path, s3_bucket_name, s3_key)
        return jsonify({'message': f'All files in {folder_path} uploaded successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/get-log-predictions-multiple-files-from-s3-bucket', methods=['POST'])
def get_predictions_multiple_files_csv_parser_xgboost_s3():
    tenant = request.headers.get('X-TenantId')
    if not tenant:
        return jsonify({'error': 'Headder Value required'}), 400
    project_id = request.json.get('project_id')
    if not project_id:
        return jsonify({'error': 'project id fields are required'}), 400
    g.tenant = tenant
    s3_client,s3_bucket_name = get_s3_connection(project_id)
    # Check if S3 is configured
    if not s3_client or not s3_bucket_name:
        return jsonify({'error': 'S3 is not configured'}), 400
    data = request.get_json()
    file_names = data.get('file_names', [])
    model_name = data.get('model', None)
    if not file_names:
        return jsonify({'message': 'File names not defined'})
    global stacked_model
    global tfid_vec
    current_model = stacked_model
    current_vectorizer = tfid_vec
    if model_name:
        match = re.search(r'_([0-9]{4}_[0-9]{2}_[0-9]{2})$', model_name)
        if match:
            model_date = match.group(1).replace('_', '-')
            loaded_model, loaded_vectorizer = load_model_and_vectorizer(model_date)
            if loaded_model is None or loaded_vectorizer is None:
                return jsonify({'message': 'Model files not found'})
            current_model = loaded_model
            current_vectorizer = loaded_vectorizer
        else:
            return jsonify({'message': 'Invalid model name format'})
    results = {}
    tmp_dir = '/tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for file_name in file_names:
        try:
            # Download the file from S3
            local_file_path = os.path.join(tmp_dir, file_name)
            s3_client.download_file(s3_bucket_name, file_name, local_file_path)
            # Process the file
            file_path = process_log(local_file_path)
            if has_more_than_one_line(file_path):
                predictions = get_predections_csv_stacked(file_path, current_model, current_vectorizer)
                if isinstance(predictions, pd.DataFrame):
                    results[file_name] = predictions.to_dict(orient='records')
                else:
                    results[file_name] = {'message': 'Load the model and the encoder'}
            else:
                results[file_name] = {'message': 'No Data'}
            # Remove the local file after processing
            os.remove(local_file_path)
        except Exception as e:
            results[file_name] = {'message': f'Error processing file: {str(e)}'}
    return jsonify(results)
def get_predections_csv_stack_list(df):
    global stacked_model
    global tfid_vec 
    # Get the name of the first column
    first_column_name = df.columns[0]
    # column = df['message']
    if stacked_model is not None and tfid_vec is not None:
        # Process the error messages
        X_test_data = df[first_column_name].apply(pre_preocess_errors)
        encoded_test_X = tfid_vec.transform(X_test_data)  
        # Get predictions
        test_predictions = stacked_model.predict(encoded_test_X) 
        # Get probability estimates
        new_probability_estimates = stacked_model.predict_proba(encoded_test_X)
        results_df = pd.DataFrame({
            'error': df[first_column_name],
            'sub_issue': test_predictions,
            'issue': None,
            'resolution': None,
            'percentage': round(np.max(new_probability_estimates) * 100, 2),
            'summary': None,
            'priority': None
        })
        for index, row in results_df.iterrows():
            results_df.at[index, 'issue'] = get_issue(row['sub_issue'])
            results_df.at[index, 'resolution'] = get_resolution(row['error'])
            results_df.at[index, 'summary'] = summarize(row['error'])
            results_df.at[index, 'priority'] = predict_priority(row['error'])
        return results_df
    else:
        return "Load the model and the encoder"
@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    columns = data.get('column_name', [])
    output_col = data.get('output_col_names', [])
    # Ensure columns is a list
    if not (isinstance(columns, list) and all(isinstance(col, str) for col in columns) and
            isinstance(output_col, list) and all(isinstance(col, str) for col in output_col)):
        return jsonify(message="Invalid data format. 'column_name' and 'output_col_names' should be lists of strings."), 400
    df = global_df
    # Ensure global_df is not None
    if df is None:
        return jsonify(message="No data available. Please upload a file first."), 400
    # Initialize an empty DataFrame to collect all error rows
    error_rows = pd.DataFrame()
    for col in columns:
        if col in df.columns:
            # Validate column data type
            if not df[col].apply(lambda x: isinstance(x, str)).all():
                return jsonify(message=f"Column '{col}' contains non-string values. Columns data invalid."), 400
            # Filter rows where the column contains 'error'
            col_errors = df[df[col].str.contains('error', case=False, na=False)]
            error_rows = pd.concat([error_rows, col_errors])
        else:
            return jsonify(message=f"Column '{col}' not found"), 400
    # Remove duplicate rows, if any
    error_rows = error_rows.drop_duplicates()
    # Check if output_col exists in the filtered DataFrame
    missing_cols = [i for i in output_col if i not in df.columns]
    if missing_cols:
        return jsonify(message=f"Columns '{missing_cols}' are not found"), 400
    # Extract only the specified columns from the filtered DataFrame
    # Check if output_col exists in the error_rows DataFrame
    missing_output_cols = [i for i in output_col if i not in error_rows.columns]
    if missing_output_cols:
        return jsonify(message=f"Columns '{missing_output_cols}' are not found in error rows"), 400
    messages = error_rows[output_col].to_dict(orient='records')
    global stacked_model
    global tfid_vec
    if stacked_model is None or tfid_vec is None:
        return jsonify({'message': 'Load the model and the encoder'}), 400
    # Convert messages to DataFrame
    df = pd.DataFrame(messages)
    # Process messages to get predictions
    predictions_df = get_predections_csv_stack_list(df)
    if isinstance(predictions_df, pd.DataFrame):
        return jsonify(predictions_df.to_dict(orient='records'))
    else:
        return jsonify({'message': 'Failed to process predictions'}), 500
##########################################################################################
# Define the /help route to list APIs and their documentation
@app.route('/help', methods=['GET'])
def routes_info():
    """
    Get a list of available routes, their HTTP methods, and docstrings.
    Request Format:
    - Method: GET
    Returns:
    - JSON response with a list of available routes, HTTP methods, and docstrings.
    """
    routes = list_routes()
    return jsonify(routes)
if __name__ == '__main__':
    app.run(port=5004,host="0.0.0.0")


Hi pasindu! 
these are the unused functions I found in app.py
 
get_extracted_errors

get_predictions_from_csv_encode

get_older_models

train_priority_model

get_predections_csv_xgboost

update_record

load_old_model_cat

routes_info

get_predictions_multiple_files_csv_parser_xgboost_s3

reset_old_model_cat

configure_s3

get_predictions_csv_parser_xgboost

get_predictions_list

list_files_in_s3

update_model

schedule_job

extract

get_best_resolution

assign_work_endpoint

get_models

get_sub_issue_xgboost

start_retrain

re_train_models

get_predictions_xgboost

load_old_resolution_model

get_resolution_data

create_issue

upload_file

upload_files

reset_resolution_model_cat

get_cat_data

get_predections_csv_xgboost_list

process_json

update_resolution_model

get_predictions_multiple_files_csv_parser_xgboost

upload_fixed_folder_to_s3 
 
I ran with vulture module, it shows so many functions also 
 
thanks there is one more thing
 
Yes pasindu, tell me
 
{

    "baseUrl": "http://10.64.65.95:5004/",

    "uploadFileEndpoint": "uploads",

    "uploadMultipleFilesEndpoint" :"upload_files",

    "analyseErrorsEndpoint": "get-log-predictions",

    "analyseErrorsInMultipleFilesEndpoint": "get-log-predictions-multiple-files",

    "editFailureDataEndpoint": "add-new-data",

    "updateFailureDataEndpoint": "update-data",

    "jiraTicketEndPoint": "jira-create",

    "getModelsListEndpoint" : "get-models",

    "getFilesFromS3" : "list_files",

    "connectToS3" : "s3_config",

    "getPredictionsFromS3BucketFiles" : "get-log-predictions-multiple-files-from-s3-bucket",

    "getMachineKeyEndpoint" : "machine_id",

    "decryptLicenseEndpoint" : "decrypt_license",

    "userRegisterEndpoint" : "register",

    "userLoginEndpoint" : "login",

    "userLogoutEndpoint" : "logout",

    "extractCSVEndpoint" : "extract"

}

 
 
there are the current endpoints in use forget about anyhtin log in loogout and stuff ig those endpoints only in use fugre out what are the enpoints that are not in use then there will be more un used methods 
 
alos in the functions list do not consder end points for exmaple 
 
get_extracted_errors
 
this is a endpoint so yes this will not be called with in the file it will call as a api so those mayeb in use 
 
cross check with the list 
 


