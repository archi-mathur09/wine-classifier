import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model_file = 'loan_classifier_rf.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Title and image
st.title('Loan Approval Prediction')
st.image('https://lendingplate.com/blog/wp-content/uploads/2023/08/Instant-Loan-Approval.png', width=500)    

st.sidebar.header('Input Parameters')

def input_features():
    loan_id=st.sidebar.number_input('loan id',min_value=1,max_value=4269,value=1)
    no_of_dependents=st.sidebar.number_input('no of dependents',min_value=0,max_value=5,value=0)
    education=st.sidebar.radio('education',options=['Graduate','Not Graduate'])
    self_employed=st.sidebar.radio('self employed',options=['yes','No'])
    income_annum=st.sidebar.number_input('income annum',min_value=200000,max_value=9900000,value=9900000)
    loan_amount=st.sidebar.number_input('loan amount',min_value=300000,max_value=39500000,value=39500000)
    loan_term=st.sidebar.number_input('loan term',min_value=2,max_value=20,value=2)
    cibil_score=st.sidebar.number_input('cibil score',min_value=300,max_value=900,value=300)
    residential_assets_value=st.sidebar.number_input('residential assets value',min_value=100000,max_value=29100000,value=29100000)
    commercial_assets_value=st.sidebar.number_input('commercial assets value',min_value=0,max_value=19400000,value=19400000)
    luxury_assets_value=st.sidebar.number_input('luxury assets value',min_value=0,max_value=39200000,value=39200000)
    bank_asset_value =st.sidebar.number_input('bank asset value',min_value=0,max_value=14700000)

    data={
        'loan id':loan_id,
        'no of dependents':no_of_dependents,
        'education':education,
        'self employed':self_employed,
        'income annum':income_annum,
        'loan amount':loan_amount,
        'loan term':loan_term,
        'cibil score':cibil_score,
        'residential assets value':residential_assets_value,
        'commercial assets value':commercial_assets_value,
        'luxury assets value':luxury_assets_value,
        'bank asset value':bank_asset_value 


    }
    # Convert the dictionary into a DataFrame
    features = pd.DataFrame(data, index=[0])  #coverting dictionary into dataframe using pandas

    # Encoding categorical features
    label_encoder = LabelEncoder()
    categorical_columns = ['education', 'self_employed']  # Use 'Product ID' to match the model

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