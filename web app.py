# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:05:29 2021

@author: harshit saxena
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users/harshit saxena/machine learning/diabetes pridiction project/trained_model.sav', 'rb'))

# creating a function for prediction

def diabetes_prediction(input_data):

    #changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    #standardize the input data
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0) :
      return"the person is non diabetic"
    else:
      return"the person is diabetic"
      
def main():
    #giving a title
    st.title('diabetes prediction web app')
    
    Pregnancies = st.text_input("number of pregnancies")
    Glucose = st.text_input("glucose level")
    BloodPressure = st.text_input("blood pressure value")
    SkinThickness = st.text_input("skin thickness value")
    Insulin = st.text_input("Insulin value")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input(" Diabetes pedigree value")
    Age = st.text_input("Age")
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    