# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:37:30 2023

@author: saimo
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Framingham dataset 
data = pd.read_csv('Framingham_app/framingham.csv')

# Streamlit web app title
st.title('Framingham Heart Study Dataset Viewer')

# Sidebar to filter data
st.sidebar.header('Data Filters')
age_filter = st.sidebar.slider('Filter by Age', min_value=int(data['age'].min()), max_value=int(data['age'].max()))
sex_filter = st.sidebar.selectbox('Filter by Sex', ['All'] + data['sex'].unique())

# Apply filters
filtered_data = data[(data['age'] <= age_filter) & (data['sex'] if sex_filter == 'All' else (data['sex'] == sex_filter))]

# Display dataset size
st.write("### Filtered Data Size")
st.write(f"Number of Rows: {len(filtered_data)}")

# Show a bar chart of age distribution
st.write("### Age Distribution")
age_dist = sns.histplot(data=filtered_data, x='age', kde=True)
st.pyplot(age_dist)

# Interactive Scatter Plot
st.write("### Interactive Scatter Plot")
x_axis = st.selectbox('X-Axis:', data.columns)
y_axis = st.selectbox('Y-Axis:', data.columns)
scatter_plot = st.scatter_chart(data=filtered_data, x=x_axis, y=y_axis)
st.pyplot(scatter_plot)

# Correlation Heatmap
st.write("### Correlation Heatmap")
corr = filtered_data.corr()
heatmap = sns.heatmap(corr, annot=True)
st.pyplot(heatmap)



