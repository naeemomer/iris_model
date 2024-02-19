#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import joblib
import numpy as np
import pickle
# Load the trained model

model_pkl_file = "iris_classifier_model.pkl"  
# load model from pickle file
with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)

# evaluate model 
# y_predict = model.predict(X_test)

# model_iris = joblib.load('Iris_Model.pkl', 'wb')
# # Assuming 'model' is your trained model object
# with open('Iris_Model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# filename='Crude_Allocation_IO.sav'
# model_etr = pickle.load(open('Crude_Allocation_IO.sav','rb'))

# # Create Streamlit app

st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into setosa, versicolor, virginica')

st.header("Plant Features")

# st.title('Iris Classifier')
sepal_length = st.number_input('Sepal Length', min_value=4.0, max_value=8.0, step=0.1, value=5.0)
sepal_width = st.number_input('Sepal Width', min_value=2.0, max_value=5.0, step=0.1, value=3.0)
petal_length = st.number_input('Petal Length', min_value=1.0, max_value=7.0, step=0.1, value=4.0)
petal_width = st.number_input('Petal Width', min_value=0.1, max_value=3.0, step=0.1, value=1.0)

if st.button('Predict'):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write('Predicted Class:', prediction[0])

# #----------------------------------------------------------------------------
# st.title('Classifying Iris Flowers')
# st.markdown('Toy model to play to classify iris flowers into setosa, versicolor, virginica')

# st.header("Plant Features")
# col1, col2 = st.columns(2)
# with col1:
#     st.text("Sepal characteristics")
#     sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
#     sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
# with col2:
#     st.text("Pepal characteristics")
#     petal_l = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
#     petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

# if st.button('Predict'):
#     # Prepare input data
#     input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
#     # Make prediction using the loaded model
#     prediction = model.predict(input_data)
    
#     # Display prediction
#     st.write('Predicted Class:', prediction[0])


# In[12]:


# model_iris


# In[ ]:





# In[ ]:




