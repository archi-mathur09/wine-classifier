import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Introduction:
# This project is a Wine Quality Prediction App that uses a machine learning model to predict the quality of wine based on various chemical properties. The app provides an interactive interface where users can input wine features such as acidity, sugar content, and alcohol level to get a prediction of the wine's quality.

# Aim:
# The aim of this project is to create a simple, user-friendly web app that allows users to predict wine quality using a pre-trained Random Forest model, based on key chemical attributes of the wine.

#  output:
# 0-2: Poor quality wines, likely unpalatable.
# 3-5: Average quality wines, acceptable but not outstanding.
# 6-8: Good to very good quality wines, enjoyable and well-balanced.
# 9-10: Exceptional quality wines, high-end selections often favored by connoisseurs.


#loading our pre-trained model
model_file='wine_classifier_rf.pkl'
with open(model_file,'rb') as file:
    model=pickle.load(file)

#title
st.title("Wine Prediction")    
st.image("https://raw.githubusercontent.com/Masterx-AI/Project_Wine_Quality_Investigation/main/wq.jpg")

#sidebar for user input parameters
st.sidebar.header('Input Parameters')

#Function to get user input
#using number_input to take only number as input
#min_value=min value to be taken
#max_value=max value to be taken
#value=bydefault value
def user_input_features():
    fixed_acidity = st.sidebar.number_input('Fixed Acidity', min_value=4.6,max_value=15.9, value=8.0)
    volatile_acidity = st.sidebar.number_input('Volatile Acidity', min_value=0.1, max_value=1.5, value=0.5)
    citric_acid = st.sidebar.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.5)
    residual_sugar = st.sidebar.number_input('Residual Sugar', min_value=0.9, max_value=15.5, value=2.0)
    chlorides = st.sidebar.number_input('Chlorides', min_value=0.0, max_value=0.6, value=0.08)
    free_sulfur_dioxide = st.sidebar.number_input('Free Sulfur Dioxide', min_value=1.0, max_value=68.0, value=20.0)
    total_sulfur_dioxide = st.sidebar.number_input('Total Sulfur Dioxide', min_value=6.0, max_value=289.0, value=50.0)
    density = st.sidebar.number_input('Density', min_value=0.9, max_value=1.0, value=0.995)
    pH = st.sidebar.number_input('pH', min_value=2.7, max_value=4.0, value=3.3)
    sulphates = st.sidebar.number_input('Sulphates', min_value=2.0, max_value=2.5, value=2.1)
    alcohol = st.sidebar.number_input('Alcohol', min_value=14.9, max_value=15.0, value=14.9)

    #After collecting input the input is stored in the form of dictionary
    #dictionary for all the features
    data={
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,

    }
    # Convert dictionary into DataFrame
    features = pd.DataFrame(data, index=[0])
    return features
# it stores the user input in var input_df
input_df = user_input_features()

# Display the input dataframe
st.subheader('User Input Parameters')
st.write(input_df) #so that user can see its input parameters

# Predict button
if st.button("Predict Cancer"):
    # Predict using the loaded model
    prediction = model.predict(input_df)    #The model.predict() function uses the trained model to predict wine quality based on the user inputs
    
    # Display prediction result
    st.subheader('Prediction')
    st.write(f'Predicted Wine Quality: {prediction[0]}')
    
    
