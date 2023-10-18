# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:37:30 2023

@author: saimo
"""

import streamlit as st
import pandas as pd
import altair as alt

import streamlit as st




import streamlit as st

# Add custom CSS to change background color to black
st.markdown(
    """
    <style>
    body {
        background-color: #000000;  /* Black color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add title to the Streamlit app
st.title("Black Background Streamlit App")

# Sidebar to select the page
page = st.sidebar.selectbox("Select a page", ["Page 1", "Page 2"])

# Page 1 content
if page == "Page 1":
    st.subheader("Page 1")
    st.write("This is the content for Page 1.")

# Page 2 content
if page == "Page 2":
    st.subheader("Page 2")
    st.write("This is the content for Page 2.")









# Load the Framingham dataset (you'll need to replace 'framingham.csv' with the actual dataset file path)
data = pd.read_csv('Framingham_app/framingham.csv')

# Streamlit web app title
st.title('Framingham Heart Study Dataset Viewer')

# Display the dataset
if st.checkbox('Show The Framingham Data'):
    #st.write(pd.DataFrame(data, columns=df.columns))
    st.write(data.columns)
    st.write('Basic Framingham Dataset Information:')
    st.write(f'Total Number of Samples: {data.shape[0]}')
    st.write(f'Number of Features: {data.shape[1]}')

# Display dataset summary statistics
st.write("### Summary Statistics")
st.write(data.describe())

# Display dataset
# st.write("### Framingham Heart Study Dataset")
# st.write(data)

# Sidebar to filter data
st.header('Data Filters')
age_filter = st.slider('Filter by Age', min_value=int(data['age'].min()), max_value=int(data['age'].max()))

# Apply filters
filtered_data = data[data['age'] <= age_filter]

# Display filtered data
st.write("### Filtered Data")
st.write(filtered_data)

# Show an Altair plot of age distribution
st.write("### Age Distribution")
age_chart = alt.Chart(filtered_data).mark_bar().encode(
    x=alt.X('age:Q', bin=True),
    y='count()',
    tooltip=['age:Q', 'count()']
).interactive()
st.altair_chart(age_chart)

# Interactive scatter plot
st.write("### Interactive Scatter Plot")
x_column = st.selectbox("X-axis", filtered_data.columns)
y_column = st.selectbox("Y-axis", filtered_data.columns)
scatter_chart = alt.Chart(filtered_data).mark_circle().encode(
    x=x_column,
    y=y_column,
    tooltip=[x_column, y_column]
).interactive()
st.altair_chart(scatter_chart)

# Add more features and interactions as needed




