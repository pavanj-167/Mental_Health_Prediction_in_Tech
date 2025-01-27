import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "/workspaces/Mental_Health_Prediction_in_Tech/models/random_forest_model.pkl"

def prediction():
    st.write("Provide the details below to predict if treatment is needed.")
    
    gender_map = {"Male": 1, "Female": 0}
    self_employed_map = {"Yes": 1, "No": 0}
    family_history_map = {"Yes": 1, "No": 0}
    work_interfere_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    no_employees_map = {
        "1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3, 
        "500-1000": 4, "More than 1000": 5
    }
    remote_work_map = {"Yes": 1, "No": 0}
    tech_company_map = {"Yes": 1, "No": 0}
    benefits_map = {"No": 0, "Yes": 1, "Don't know": 2}
    care_options_map = {"No": 0, "Not sure": 1, "Yes": 2}
    wellness_program_map = {"No": 0, "Don't know": 1, "Yes": 2}
    seek_help_map = {"No": 0, "Don't know": 1, "Yes": 2}
    anonymity_map = {"No": 0, "Don't know": 1, "Yes": 2}
    leave_map = {
        "Very difficult": 0, "Somewhat difficult": 1, 
        "Don't know": 2, "Somewhat easy": 3, "Very easy": 4
    }
    mental_health_consequence_map = {"No": 0, "Maybe": 1, "Yes": 2}
    phys_health_consequence_map = {"No": 0, "Maybe": 1, "Yes": 2}
    coworkers_map = {"No": 0, "Some of them": 1, "Yes": 2}
    supervisor_map = {"No": 0, "Some of them": 1, "Yes": 2}
    mental_health_interview_map = {"No": 0, "Maybe": 1, "Yes": 2}
    phys_health_interview_map = {"No": 0, "Maybe": 1, "Yes": 2}
    mental_vs_physical_map = {"Don't know": 0, "No": 1, "Yes": 2}
    obs_consequence_map = {"No": 0, "Yes": 1}

    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    self_employed = st.selectbox("Are you self-employed?", list(self_employed_map.keys()))
    family_history = st.selectbox("Family history of mental illness?", list(family_history_map.keys()))
    work_interfere = st.selectbox("Work interference", list(work_interfere_map.keys()))
    no_employees = st.selectbox("Number of employees", list(no_employees_map.keys()))
    remote_work = st.selectbox("Do you work remotely?", list(remote_work_map.keys()))
    tech_company = st.selectbox("Do you work in a tech company?", list(tech_company_map.keys()))
    benefits = st.selectbox("Do you receive mental health benefits?", list(benefits_map.keys()))
    care_options = st.selectbox("Are care options available?", list(care_options_map.keys()))
    wellness_program = st.selectbox("Does your employer provide a wellness program?", list(wellness_program_map.keys()))
    seek_help = st.selectbox("Does your employer provide resources to seek help?", list(seek_help_map.keys()))
    anonymity = st.selectbox("Is anonymity protected?", list(anonymity_map.keys()))
    leave = st.selectbox("Ease of taking leave for mental health", list(leave_map.keys()))
    mental_health_consequence = st.selectbox("Fear of mental health consequences at work?", list(mental_health_consequence_map.keys()))
    phys_health_consequence = st.selectbox("Fear of physical health consequences at work?", list(phys_health_consequence_map.keys()))
    coworkers = st.selectbox("Discuss mental health with coworkers?", list(coworkers_map.keys()))
    supervisor = st.selectbox("Discuss mental health with supervisor?", list(supervisor_map.keys()))
    mental_health_interview = st.selectbox("Would you bring up mental health in an interview?", list(mental_health_interview_map.keys()))
    phys_health_interview = st.selectbox("Would you bring up physical health in an interview?", list(phys_health_interview_map.keys()))
    mental_vs_physical = st.selectbox("Is mental health as important as physical health?", list(mental_vs_physical_map.keys()))
    obs_consequence = st.selectbox("Have you observed negative consequences for discussing mental health?", list(obs_consequence_map.keys()))

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_map[gender]],
        "self_employed": [self_employed_map[self_employed]],
        "family_history": [family_history_map[family_history]],
        "work_interfere": [work_interfere_map[work_interfere]],
        "no_employees": [no_employees_map[no_employees]],
        "remote_work": [remote_work_map[remote_work]],
        "tech_company": [tech_company_map[tech_company]],
        "benefits": [benefits_map[benefits]],
        "care_options": [care_options_map[care_options]],
        "wellness_program": [wellness_program_map[wellness_program]],
        "seek_help": [seek_help_map[seek_help]],
        "anonymity": [anonymity_map[anonymity]],
        "leave": [leave_map[leave]],
        "mental_health_consequence": [mental_health_consequence_map[mental_health_consequence]],
        "phys_health_consequence": [phys_health_consequence_map[phys_health_consequence]],
        "coworkers": [coworkers_map[coworkers]],
        "supervisor": [supervisor_map[supervisor]],
        "mental_health_interview": [mental_health_interview_map[mental_health_interview]],
        "phys_health_interview": [phys_health_interview_map[phys_health_interview]],
        "mental_vs_physical": [mental_vs_physical_map[mental_vs_physical]],
        "obs_consequence": [obs_consequence_map[obs_consequence]]
    })

    if st.button("Predict"):
        loaded_model = joblib.load(MODEL_PATH)
        prediction = loaded_model.predict(input_data)
        result = "Needs Treatment" if prediction == 1 else "Does Not Need Treatment"
        st.write(f"The prediction is: **{result}**")