# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 08:44:15 2023

@author: saimo
"""

import streamlit as st
import seaborn as sns
import pandas as pd
df = pd.read_csv("path-to-the-data")
st.write("plots for the cancer dataset")
X_axis = st.text_input('input for X-axis ')
Y_axis = st.text_input('input for Y-axis')

#the plot shows the radius mean of the B and M diagnos
g = sns.catplot(x= X_axis, y= Y_axis, data=df)

st.pyplot(g)
