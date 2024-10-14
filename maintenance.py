
# Introduction:
# This project focuses on predictive maintenance for a milling machine, using a synthetic dataset containing 10,000 records and 14 features. The dataset captures various operational parameters like air temperature, process temperature, rotational speed, torque, tool wear, and different machine failure modes. It models the behavior of a real milling machine, with data representing quality variations, failure modes, and process conditions. The dataset is designed to enable machine learning techniques for predicting potential machine failures.

# Aim:
# The aim of this project is to develop a predictive maintenance model that can accurately forecast machine failures based on the given operational parameters. By identifying conditions that lead to failures, the model seeks to optimize maintenance schedules, reduce downtime, and improve the overall reliability and efficiency of the milling machine.

#Output:
# 0-Machine do not failed
# 1-Machine Failed


import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model_file = 'maintenance_classifier_lr.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Title and image
st.title('Predictive Maintenance Prediction')
st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHupGT4jzPMp7GaqyzYGqiwPgxJgxaMTyxLw&s', width=500)

st.sidebar.header('Input Parameters')

def input_features():
    UDI = st.sidebar.number_input('UDI', min_value=1, max_value=1000, value=1)
    product_ID = st.sidebar.text_input('Product ID')  # Changed to 'Product ID'
    Type = st.sidebar.radio('Type', options=["M", "L", "H"])
    Air_temperature_K = st.sidebar.number_input('Air Temperature [K]', min_value=295.3, max_value=304.5, value=298.0)
    Process_temperature_K = st.sidebar.number_input('Process Temperature [K]', min_value=305.7, max_value=313.8, value=306.0)
    Rotational_speed_rpm = st.sidebar.number_input('Rotational Speed [rpm]', min_value=1168, max_value=2886, value=1168)
    Torque_Nm = st.sidebar.number_input('Torque [Nm]', min_value=3.8, max_value=76.6, value=5.0)
    Tool_wear_min = st.sidebar.number_input('Tool Wear [min]', min_value=0, max_value=253, value=5)
    Machine_failure = st.sidebar.radio('Machine Failure', options=["0", "1"])
    TWF = st.sidebar.radio('TWF', options=["0", "1"])
    HDF = st.sidebar.radio('HDF', options=["0", "1"])
    PWF = st.sidebar.radio('PWF', options=["0", "1"])
    OSF = st.sidebar.radio('OSF', options=["0", "1"])

    # Create a dictionary with the correct feature names
    data = {
        'UDI': UDI,
        'Product ID': product_ID,  # Correct the name to 'Product ID'
        'Type': Type,
        'Air temperature [K]': Air_temperature_K,
        'Process temperature [K]': Process_temperature_K,
        'Rotational speed [rpm]': Rotational_speed_rpm,
        'Torque [Nm]': Torque_Nm,
        'Tool wear [min]': Tool_wear_min,
        'Machine failure': Machine_failure,
        'TWF': TWF,
        'HDF': HDF,
        'PWF': PWF,
        'OSF': OSF
    }

    # Convert the dictionary into a DataFrame
    features = pd.DataFrame(data, index=[0])  #coverting dictionary into dataframe using pandas

    # Encoding categorical features
    label_encoder = LabelEncoder()
    categorical_columns = ['Product ID', 'Type']  # Use 'Product ID' to match the model

    for column in categorical_columns:
        if column in features:  # Check if the column exists in the DataFrame
            features[column] = label_encoder.fit_transform(features[column])

    return features

# Get user input
input_df = input_features()

# Display the input DataFrame
st.subheader('User Input Parameters')
st.write(input_df)

# Define a mapping for predictions
prediction_mapping = {0: 'Machine does not Failed', 1: 'Machine Fail'}

# Predict button
if st.button("Predict"):
    
        # Predict using the loaded model
        prediction = model.predict(input_df)

        # Map numerical prediction to human-readable labels
        predicted_label = prediction_mapping[prediction[0]]

        # Display prediction result
        st.subheader('Prediction')
        st.write(f'The machine is predicted to be: {predicted_label}')
  



