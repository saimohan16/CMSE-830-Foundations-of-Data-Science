# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 22:43:33 2023

@author: saimo
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Hepatitis dataset
@st.cache  # Use Streamlit's caching for better performance
def load_data():
    column_names = [
        'Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise',
        'Anorexia', 'Liver_Big', 'Liver_Firm', 'Spleen_Palpable', 'Spiders',
        'Ascites', 'Varices', 'Bilirubin', 'Alk_Phosphate', 'SGOT', 'Albumin',
        'Protime', 'Histology'
    ]
    hepatitis_df = pd.read_csv(r"C:\Users\saimo\Downloads\hepatitis\hepatitis.data", names=column_names, na_values='?')
    return hepatitis_df

hepatitis_df = load_data()

# Streamlit app
st.title('Hepatitis Dataset Exploration')

# Display the dataset
st.write("### Hepatitis Dataset")
st.write(hepatitis_df)

# Summary statistics
st.write("### Summary Statistics")
st.write(hepatitis_df.describe())

# Data visualization (Seaborn plots)
st.write("### Data Visualization")

# Example 1: Pairplot (scatter plots and histograms)
st.write("#### Pairplot")
sns.pairplot(hepatitis_df, hue='Class')
st.pyplot()

# Example 2: Correlation Heatmap
st.write("#### Correlation Heatmap")
correlation_matrix = hepatitis_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
st.pyplot()

# Add more Seaborn visualizations as needed

# Footer
st.write("Created with Streamlit")

