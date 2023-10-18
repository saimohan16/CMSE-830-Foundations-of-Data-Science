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
@st.cache
def load_data():
    data = pd.read_csv('Framingham_app/framingham.csv')  # Replace with the actual path to your dataset
    return data

data = load_data()

# Create tabs for the two pages
tabs = st.beta_container()

with tabs:
    selected_tab = st.radio("Select a page", ("Data Information", "Data Plots"))
    
    # Page 1: Information about the data
    if selected_tab == "Data Information":
        st.title("Framingham Heart Study Data")
        st.write("This page provides information about the Framingham dataset.")
        st.write("The Framingham Heart Study dataset contains information about heart disease risk factors and outcomes.")
    
    # Page 2: Plots
    if selected_tab == "Data Plots":
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

        
        st.title("Data Visualization")
        st.write("This page displays plots and visualizations of the Framingham dataset.")
        
        # Example: Plot a histogram of Age
        st.subheader("Age Distribution")
        plt.figure(figsize=(8, 6))
        sns.histplot(data['age'], bins=20, kde=True)
        st.pyplot()
        
        # Add more plots and visualizations as needed

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
#st.write("### Interactive Scatter Plot")
#x_column = st.selectbox("X-axis", filtered_data.columns)
#y_column = st.selectbox("Y-axis", filtered_data.columns)
#scatter_chart = alt.Chart(filtered_data).mark_circle().encode(
#   x=x_column,
#    y=y_column,
#    tooltip=[x_column, y_column]
#).interactive()
#st.altair_chart(scatter_chart)

# Add more features and interactions as needed




