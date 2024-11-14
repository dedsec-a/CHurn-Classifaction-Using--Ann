import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler , OneHotEncoder ,LabelEncoder
import pickle
import tensorflow as tf

# Laoding the Model 
model = tf.keras.models.load_model('model.h5')

# Loading the Encoder Files 
with open('label_encoder_gender.pkl' , 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('One_Hot_Encoder.pkl' , 'rb') as file:
    oh = pickle.load(file)
    
with open('standard_scaller.pkl' , 'rb') as file:
    standard_scaller = pickle.load(file)


# StreamLit APP
st.title("Customre Churn Predection ")

# User Input 
geography = st.selectbox('Geography', oh.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])



# Making a Dataframe For Data inout
user_input = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


user_input = pd.DataFrame(user_input)

geo_encoded = oh.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=oh.get_feature_names_out(['Geography']))



user_input = pd.concat([user_input.reset_index(drop = True), geo_encoded_df] , axis= 1)



# Scaling the Data 
input_data_scaled = standard_scaller.transform(user_input)


# Making Predections 
predection = model.predict(input_data_scaled)
predection_proba = predection[0][0]


st.write("Churn Probablity" , predection_proba)

if predection_proba >0.5 :
    st.write("The Customer is Likely to Churn")
else :
    st.write("The customer is Not Likey to Churn")