import streamlit as st
import pandas as pd

x = st.slider(label='Helo')
st.write(x, 'square is ', x*x)

df = pd.DataFrame({'col1': [1,2,3]})
df  # <-- Draw the dataframe

x = 10
'x', x  #

