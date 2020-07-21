# Visualising the Model predictions with the dynamic inputs

# Fixme: Change port no
# Todo: Add Data uploading [feature]
# Todo: Add Data Exploration [feature]
# Todo: Live model training [feature]
# Todo: Add more charts for Model outputs[Enhancement]
# Todo: Automatic widget finder [Enhancement]

import streamlit as st
import json
import pandas as pd

import utils as ut
import model_prediction as mp

# Contains factor details like min, max and values(For categories)
with open("data/factors.json") as file:
    factors = json.load(file)

factors = factors['factors']

sex_values = [values['range'] for values in factors if values['factor_name'] == 'Sex']
housing_values = [values['range'] for values in factors if values['factor_name'] == 'Housing']
savings_accounts = [values['range'] for values in factors if values['factor_name'] == 'Saving accounts']
checking_account = [values['range'] for values in factors if values['factor_name'] == 'Checking account']
purpose = [values['range'] for values in factors if values['factor_name'] == 'Purpose']
job = [values['range'] for values in factors if values['factor_name'] == 'Job']
credit_amount = [values['range'] for values in factors if values['factor_name'] == 'Credit amount']
age = [values['range'] for values in factors if values['factor_name'] == 'Age']
duration = [values['range'] for values in factors if values['factor_name'] == 'Duration in Months']

# Creating widgets
sex = st.sidebar.selectbox('Sex', sex_values[0])
housing = st.sidebar.selectbox('Housing', housing_values[0])
saving_accounts = st.sidebar.selectbox('Saving account', savings_accounts[0])
checking_account = st.sidebar.selectbox('Checking account', checking_account[0])
purpose = st.sidebar.selectbox('Purpose', purpose[0])
job = st.sidebar.selectbox('Job', job[0])
ca = st.sidebar.slider(label='Credit amount', min_value=credit_amount[0][0], max_value=credit_amount[0][1], step=
               [values['interval'] for values in factors if values['factor_name'] == 'Credit amount'][0])
age = st.sidebar.slider(label='Age', min_value=age[0][0], max_value=age[0][1], step=
               [values['interval'] for values in factors if values['factor_name'] == 'Age'][0])
duration = st.sidebar.slider(label='Duration in Months', min_value=duration[0][0], max_value=duration[0][1], step=
               [values['interval'] for values in factors if values['factor_name'] == 'Duration in Months'][0])

# Organises the input factors for the prediction
factors = ut.factors_organiser(age, sex, job, housing, saving_accounts, checking_account,
                                   ca, duration, purpose)
# Model prediction
out = mp.prob_prdictor(factors)

out = pd.DataFrame.from_dict(out, orient='index')
st.bar_chart(data=out)




