#Introduction-
# Breast cancer is a leading cause of cancer-related deaths among women worldwide, making early detection essential for improving survival rates. This project aims to develop a web application that utilizes a machine learning model to analyze various clinical parameters and provide accurate predictions regarding breast cancer presence. By leveraging advancements in artificial intelligence, the application seeks to assist healthcare professionals and individuals in making informed decisions about screening and diagnosis.

#Aim-
# The aim of this project is to develop an interactive web application that leverages a machine learning model to predict the likelihood of breast cancer based on user-input clinical parameters. By providing an easy-to-use interface, the application seeks to enhance early detection, support informed decision-making, and promote awareness about breast cancer.

#Output- whether a person is Alive or Dead



import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model_file = 'cancer_classifier_rf (1).pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Title and image
st.title('Breast Cancer Detection')
st.image('https://m.media-amazon.com/images/I/31XEhNthKQL._AC_UF894,1000_QL80_.jpg', width=100)

st.sidebar.header('Input Parameters')

def input_features():
    Age = st.sidebar.number_input('Age', min_value=30, max_value=69, value=40)
    Race = st.sidebar.radio("Race", options=["Black", "White"])
    Marital_Status = st.sidebar.radio("Marital Status", options=["Single", "Married", "Divorced", "Widowed"])
    T_Stage = st.sidebar.radio("T Stage ", options=["T1", "T2", "T3", "T4"])
    N_Stage = st.sidebar.radio("N Stage", options=["N1", "N2", "N3"])
    stage_6th = st.sidebar.radio("6Th Stage", options=["IIA", "IIIA", "IIB", "IIIC"])
    differentiate = st.sidebar.radio("Differentiate", options=["Well differentiated", "Moderately differentiated", "Poorly differentiated"])
    Grade = st.sidebar.number_input('Grade', min_value=1, max_value=1, value=1)
    A_Stage = st.sidebar.radio("A Stage", options=["Regional", "Distant"])
    Tumor_Size = st.sidebar.number_input('Tumor Size', min_value=1, max_value=140, value=35)
    Estrogen_Status = st.sidebar.radio("Estrogen Status", options=["Positive", "Negative"])
    Progesterone_Status = st.sidebar.radio("Progesterone Status", options=["Positive", "Negative"])
    Regional_Node_Examined = st.sidebar.number_input('Regional Node Examined', min_value=1, max_value=61, value=35)
    Reginol_Node_Positive = st.sidebar.number_input('Reginol Node Positive', min_value=1, max_value=46, value=31)
    Survival_Months = st.sidebar.number_input('Survival Months', min_value=1, max_value=107, value=60)

    data = {
        'Age': Age,
        'Race': Race,
        'Marital Status': Marital_Status,
        'T Stage ': T_Stage,
        'N Stage': N_Stage,
        '6th Stage': stage_6th,
        'differentiate': differentiate,
        'Grade': Grade,
        'A Stage': A_Stage,
        'Tumor Size': Tumor_Size,
        'Estrogen Status': Estrogen_Status,
        'Progesterone Status': Progesterone_Status,
        'Regional Node Examined': Regional_Node_Examined,
        'Reginol Node Positive': Reginol_Node_Positive,
        'Survival Months': Survival_Months
    }

    # Convert dictionary into DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Encoding categorical features
    label_encoder = LabelEncoder()
    categorical_columns = ['Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage', 
                           'differentiate', 'A Stage', 'Estrogen Status', 'Progesterone Status']
    
    for column in categorical_columns:
        features[column] = label_encoder.fit_transform(features[column])

    return features

# Get user input
input_df = input_features()  

# Display the input DataFrame
st.subheader('User Input Parameters')
st.write(input_df)

# Define a mapping for predictions
prediction_mapping = {0: 'Dead', 1: 'Alive'}

# Predict button
if st.button("Predict"):
    # Predict using the loaded model
    prediction = model.predict(input_df)

    # Map numerical prediction to human-readable labels
    predicted_label = prediction_mapping[prediction[0]]

    # Display prediction result
    st.subheader('Prediction')
    st.write(f'A Person is : {predicted_label}')
