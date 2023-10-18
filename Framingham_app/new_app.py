"""
Created on Fri Oct 13 20:37:30 2023

@author: saimo
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from PIL import Image
import plotly.express as px
import hiplot as hip

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the Framingham dataset
@st.cache_data
def load_data():
    data = pd.read_csv('framingham.csv')  # Replace with the actual path to your dataset
    return data

data = load_data()



tab1, tab2, tab3,tab4 = st.tabs(["About Data", "Basic information Plots", "UniVariant Analysis Plots","MultiVariant Analysis Plots"])

with tab1:
   st.title("Framingham Heart Study Data")
   image = Image.open(r"C:\Users\saimo\Desktop\Framingham_app\doctor_pointing_to_heart_graph.jpg")

   st.image(image, caption='Sunrise by the mountains')
   
   st.write("The Framingham Heart Study is a long-term, ongoing cardiovascular cohort study that began in 1948 in Framingham, Massachusetts, USA. It's one of the most well-known and influential epidemiological studies of heart disease. The study has provided valuable insights into the risk factors associated with cardiovascular disease and has helped shape our understanding of heart disease and its prevention. The Framingham dataset is a collection of data generated through this study, and it's widely used in epidemiological and public health research. The dataset contains detailed information about a variety of cardiovascular risk factors and other health-related variables for thousands of individuals. Here's an overview of the key aspects of the Framingham dataset: 1. **Study Participants**: The dataset includes information about thousands of participants from the Framingham area. It has both original and offspring cohorts, meaning it has data from different generations.2. **Data Categories**: The Framingham dataset includes information on a wide range of variables, including:- Demographic information (age, gender, etc.).- Medical history (e.g., diabetes, hypertension).")
#     )- Physical examinations (e.g., BMI, cholesterol levels).
#     - Lifestyle factors (e.g., smoking, alcohol consumption).
#     - Medication usage.
#     - Cardiovascular outcomes (e.g., heart disease, stroke).

# 3. **Longitudinal Data**: The dataset is collected over a long period, spanning multiple generations. This longitudinal data allows researchers to study the development of heart disease over time and assess the impact of various risk factors.

# 4. **Public Health Impact**: The Framingham Heart Study has led to the identification of several major risk factors for cardiovascular disease, including high blood pressure, high cholesterol, smoking, obesity, and diabetes. It has also contributed to the development of risk prediction models like the Framingham Risk Score.

# 5. **Research and Analysis**: Researchers use the Framingham dataset to conduct epidemiological studies, investigate the impact of different factors on heart health, and develop interventions and strategies to prevent cardiovascular diseases.

# 6. **Data Availability**: The dataset is often used for teaching, research, and public health purposes. It's publicly available, and you can find versions of it from various sources.

# Please note that there are different versions of the Framingham dataset, and the specific variables and details may vary between them. Researchers and analysts typically choose the version that best suits their research needs.

# If you plan to work with the Framingham dataset, you should ensure you have access to the specific version of the dataset you need and review any associated documentation to understand the variable definitions and data structure for that version.")
   
   col1,col2 = st.columns(2)
   with col1:
    on = st.toggle('feature list of the dataset')
    if on:
        #st.write('Feature activated!')
        k = list(data.columns)
        st.write(k)
        st.write('Basic Framingham Dataset Information:')
        st.write(f'Total Number of Samples: {data.shape[0]}')
        st.write(f'Number of Features: {data.shape[1]}')
    with col2:
        on1 = st.toggle('summary statistics of the dataset')
        
        if on1:
            st.write("### Summary Statistics")
            st.write(data.describe())
   

with tab2:
    st.header('Data Filters')
    age_filter = st.slider('Filter by Age', min_value=int(data['age'].min()), max_value=int(data['age'].max()))
    filtered_data = data[data['age'] <= age_filter]
    # st.write("### Filtered Data")
    # st.write(filtered_data)
    # Show an Altair plot of age distribution
    st.write("### Age Distribution")
    age_chart = alt.Chart(filtered_data).mark_bar().encode(
        x=alt.X('age:Q', bin=True),
        y='count()',
        tooltip=['age:Q', 'count()']
    ).interactive()
    st.altair_chart(age_chart)
    st.write("### Filtered Data")
    st.write(filtered_data)
    
    # Interactive scatter plot use it in the 3rd tab. 
    # st.write("### Interactive Scatter Plot")
    # x_column = st.selectbox("X-axis", filtered_data.columns)
    # y_column = st.selectbox("Y-axis", filtered_data.columns)
    # scatter_chart = alt.Chart(filtered_data).mark_circle().encode(
    #     x=x_column,
    #     y=y_column,
    #     tooltip=[x_column, y_column]
    # ).interactive()
    # st.altair_chart(scatter_chart)
    
    
#     missing_data = data.isnull()

# # Use Seaborn to create a heatmap
#     fig = plt.figure(figsize=(10, 6))
#     sns.heatmap(missing_data, cbar=False, cmap='viridis')
#     plt.title('Missing Data in Framingham Dataset')
#     st.pyplot(fig)
    
    missing_values_count = data.isnull().sum()
    plt.figure(figsize=(10, 6))
    missing_values_count.plot(kind='bar', color='skyblue')
    plt.title("Missing Values by Attribute")
    plt.xlabel("Attributes")
    plt.ylabel("Count of Missing Values")
    plt.xticks(rotation=45)
    st.pyplot()
    
    numeric_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(exclude=['number']).columns

# Impute missing values based on data type
    imputed_data = data.copy()

    for col in numeric_columns:
        imputed_data[col].fillna(imputed_data[col].mean(), inplace=True)

    for col in categorical_columns:
        imputed_data[col].fillna(imputed_data[col].mode().iloc[0], inplace=True)

# Display the original and imputed data
    st.write("Original Data:")
    st.write(data)
    st.write("Imputed Data (Mean for Numeric, Mode for Categorical):")
    st.write(imputed_data)
    st.write(imputed_data.describe())
#framingham.csv" with the actual path to your Framingham dataset. This code handles missing values according to the data type of the attribute and provides an imputed dataset for further analysis or visualization.






    
    st.title("Data Visualization")
    st.write("This page displays plots and visualizations of the Framingham dataset.")
    
    # # Example: Plot a histogram of Age
    # st.subheader("Age Distribution")
    # plt.figure(figsize=(8, 6))
    # sns.histplot(data['age'], bins=20, kde=True)
    # st.pyplot()
    # st.header("A dog")
    # st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
    data = load_data()

# Basic EDA plots for categorical variables using interactive violin plots
    st.title("Violin Plots for Categorical Variables in Framingham Dataset")

# Define the categorical variables to be visualized
    categorical_columns = ["currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp","male"]
    selected_variable = st.selectbox("Select a Categorical Variable", categorical_columns)

    fig = px.violin(data, x=selected_variable, y="age", box=True, points="all", title=f"Interactive Violin Plot for {selected_variable} vs Age")
    st.plotly_chart(fig)
    
    
    
    selected_variable = st.radio("Select a Categorical Variable", categorical_columns)

# Create box plots for the selected variable with respect to CHD

    fig = px.box(data, x=selected_variable, y="age", color="TenYearCHD",
              labels={"age": "Age", selected_variable: selected_variable, "TenYearCHD": "CHD"})
    
    fig.update_layout(
    title=f"Interactive Box Plot for {selected_variable} with Respect to CHD",
    xaxis_title='',
    yaxis_title='Age',
    showlegend=True,
)
    st.plotly_chart(fig)

# Create interactive violin plots for categorical variables
   # Create interactive violin plot for the selected variable

    # for col in categorical_columns:
    #     fig = px.violin(data, x=col, y="age", box=True, points="all", title=f"Interactive Violin Plot for {col} vs Age")
    #     st.plotly_chart(fig)

    st.title("KDE Plots for the Attributes with Respect to CHD")
    
    # Define the numerical variables to be visualized
    numerical_columns = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
    
    # Select box to choose a numerical variable
    # selected_variable = st.selectbox("Select a Numerical Variable", numerical_columns)

# Create KDE plots for the selected numerical variable with respect to CHD




# Filter the data into two subsets based on TenYearCHD
    chd_positive = data[data['TenYearCHD'] == 1]
    chd_negative = data[data['TenYearCHD'] == 0]
    
    # List of columns to plot
    columns_to_plot = data.columns.drop('TenYearCHD')
    
    # Plot categorical features (bar plots)
    for column in columns_to_plot:
        if data[column].dtype == 'object':
            plt.figure(figsize=(8, 4))
            sns.countplot(x=column, data=data, hue='TenYearCHD')
            plt.title(f'{column} vs. Ten-Year CHD')
            plt.xticks(rotation=45)
            # plt.show()
            st.pyplot()
    
    # Plot continuous features (histograms)
    for column in columns_to_plot:
        if data[column].dtype != 'object':
            plt.figure(figsize=(8, 4))
            sns.histplot(chd_negative[column], kde=True, label='No CHD', color='blue', alpha=0.6)
            sns.histplot(chd_positive[column], kde=True, label='CHD', color='red', alpha=0.6)
            plt.title(f'{column} vs. Ten-Year CHD')
            plt.xlabel(column)
            plt.legend()
            # plt.show(
            st.pyplot()

with tab4: 
    
    
    selected_variable = st.selectbox("Select a Numerical Variable", data.select_dtypes(include=['number']).columns)
    
    # Create an interactive scatter plot showing age vs. the selected numerical variable with color-coded CHD
    st.title(f"Interactive Scatter Plot: Age vs. {selected_variable} vs. Ten-Year CHD")
    
    fig = px.scatter(data, x="age", y=selected_variable, color="TenYearCHD",
                     color_continuous_scale=["blue", "red"],
                     labels={"age": "Age", selected_variable: selected_variable, "TenYearCHD": "CHD"})
    
    fig.update_layout(
        title=f"Age vs. {selected_variable} vs. Ten-Year CHD",
        xaxis_title="Age",
        yaxis_title=selected_variable
    )
    
    st.plotly_chart(fig)
    
    


    hip_exp = hip.Experiment.from_dataframe(data)
    
    # Create a Streamlit component for HiPlot
    st.title("HiPlot Visualization for Framingham Dataset")
    st_hiplot = st.empty()
    st_hiplot.hiplot(hip_exp, use_container_width=True)

    
    
    
    
#In this code, we use Plotly Express to create an interactive scatter plot. You can choose any numerical variable from the dataset using the select box, and the scatter plot will display the selected variable on the y-axis and "age" on the x-axis. The interactive plot allows you to zoom in, pan, and hover over data points for more details.








    





