import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe1.pkl', 'rb'))
df = pickle.load(open('df1.pkl', 'rb'))

st.title('House Price Predictor')

# Amount(in rupees)
amount = st.number_input('Amount (in rupees)')

# Location
location = st.selectbox('Location', df['location'].unique())

# Floor
floor = st.number_input('Floor')

# Transaction
transaction = st.selectbox('Transaction', df['Transaction'].unique())

# Furnishing
furnishing = st.selectbox('Furnishing', df['Furnishing'].unique())

# Bathroom
bathroom = st.number_input('Bathroom')

# Balcony
balcony = st.number_input('Balcony')

# BHK
bhk = st.number_input('BHK')

if st.button('Predict Price'):
    # query
    query = np.array([amount, location, floor, transaction, furnishing, bathroom, balcony, bhk])

    query = query.reshape(1, 8)
    st.title('The predicted price for house is: ' + str(int(pipe.predict(query))))