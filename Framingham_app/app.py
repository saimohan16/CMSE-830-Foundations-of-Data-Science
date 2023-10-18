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
    data = pd.read_csv('Framingham_app/framingham.csv')  # Replace with the actual path to your dataset
    return data

data = load_data()



tab1, tab2, tab3,tab4 = st.tabs(["About Data", "Basic information Plots", "UniVariant Analysis Plots","MultiVariant Analysis Plots"])

with tab1:
   st.title("Framingham Heart Study Data")
   image = Image.open('Framingham_app/doctor_pointing_to_heart_graph.jpg')

   st.image(image, caption='Image')
   
   st.title("About the DATA")
   st.write("The Framingham Heart Study is a long-term, ongoing cardiovascular cohort study that began in 1948 in Framingham, Massachusetts, USA. It's one of the most well-known and influential epidemiological studies of heart disease. The study has provided valuable insights into the risk factors associated with cardiovascular disease and has helped shape our understanding of heart disease and its prevention. The Framingham dataset is a collection of data generated through this study, and it's widely used in epidemiological and public health research.")
   st.write("The dataset contains detailed information about a variety of cardiovascular risk factors and other health-related variables for thousands of individuals. Here's an overview of the key aspects of the Framingham dataset: 1. **Study Participants**: The dataset includes information about thousands of participants from the Framingham area. It has both original and offspring cohorts, meaning it has data from different generations.2. **Data Categories**: The Framingham dataset includes information on a wide range of variables, including:- Demographic information (age, gender, etc.).- Medical history (e.g., diabetes, hypertension).")
   
   st.title("About this WebAPP")
   
   st.write("The web app is designed to provide an interactive and visually engaging platform for exploring and visualizing data from the Framingham Heart Study dataset. Users can interact with the app to:")

   st.write("1. Visualize data relationships: Explore relationships between various attributes and Ten-Year Coronary Heart Disease (CHD) status through interactive scatter plots.")

   st.write("2. Understand CHD distribution: View the proportion of CHD cases versus no CHD cases using an interactive pie chart.")

   st.write("3. Interactive 3D visualization: Discover how age, cigarettes per day, and systolic blood pressure relate to Ten-Year CHD using an interactive 3D scatter plot.")

   st.write("4. Missing data analysis: Visualize missing data patterns using heatmaps and bar plots for a comprehensive data overview.")

   st.write("5. Customizable exploration: Users can customize their exploration by selecting attributes and visualizations through drop-down menus and select boxes.")

   st.write("This web app empowers users to gain insights and understand the Framingham dataset visually, enhancing the process of data exploration and analysis.")

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
    
    chd_counts = data['TenYearCHD'].value_counts()
    chd_proportion = chd_counts / chd_counts.sum()

    st.title("Pie Chart: Proportion of CHD vs. No CHD")
    
    fig = px.pie(chd_proportion, values=chd_proportion, names=chd_proportion.index,
                 labels={'index': 'CHD Status'}, title="Proportion of CHD vs. No CHD")
    
    st.plotly_chart(fig)
    
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

    st.title ("Data Before Handling Missing Values")
    imputed_data = data.copy()
    
    col1,col2 = st.columns(2)
    
    with col1:
     on = st.toggle('feature list of the dataset',['features'])
     if on:
         st.write("Original Data:")
         st.write(data)
     else:
         col1.empty()
        #st.write('Feature activated!')
        # k = list(data.columns)
        # st.write(k)
        # st.write('Basic Framingham Dataset Information:')
        # st.write(f'Total Number of Samples: {data.shape[0]}')
        # st.write(f'Number of Features: {data.shape[1]}')

    with col2:
        on1 = st.toggle('summary statistics of the dataset',['summary'])
        
        if on1:
            st.write("### Summary Statistics")
            st.write(data.describe())





    for col in numeric_columns:
        imputed_data[col].fillna(imputed_data[col].mean(), inplace=True)

    for col in categorical_columns:
        imputed_data[col].fillna(imputed_data[col].mode().iloc[0], inplace=True)


    st.title("Imputed Data (Mean for Numeric, Mode for Categorical):")
    st.write(imputed_data.describe())
#framingham.csv" with the actual path to your Framingham dataset. This code handles missing values according to the data type of the attribute and provides an imputed dataset for further analysis or visualization.


    # chd_counts = data['TenYearCHD'].value_counts()
    # chd_proportion = chd_counts / chd_counts.sum()

    # st.title("Interactive Pie Chart: Proportion of CHD vs. No CHD")
    
    # fig = px.pie(chd_proportion, values=chd_proportion, names=chd_proportion.index,
    #              labels={'index': 'CHD Status'}, title="Proportion of CHD vs. No CHD")
    
    # st.plotly_chart(fig)

    
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
    
    st.write("Univariate analysis : Here we focus on examining individual variables one at a time. In the context of the Framingham Heart Study dataset, univariate analysis involves studying the relationship between each individual feature (independent variable) and the target variable 'TenYearCHD' (Coronary Heart Disease) to understand how each feature influences the presence of CHD. Univariate analysis helps identify which features have a significant impact on CHD.")

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
    
    st.title("Multivariant Analyis")
    
    st.write("Here we examing the relationships between multiple variables simultaneously. In the context of the Framingham Heart Study dataset, you can perform multivariate analysis to understand how combinations of features (independent variables) collectively influence the presence of Ten-Year Coronary Heart Disease (CHD) represented by the 'TenYearCHD' target variable. ")
    
    
    selected_variable = st.selectbox("Select a Numerical Variable", data.select_dtypes(include=['number']).columns)
    
    #st.write("suggested attributes for the scatter plot are")
    
    non_categorical_columns = data.select_dtypes(exclude=['object']).columns

    # Print the column names that do not have categorical values
    numerical_columns = ["cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
    
    st.write ("Suggested Attributes")
    st.write(numerical_columns)
    
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
    
    

    st.title("Interactive 3D Scatter Plot: Age, Cigarettes Per Day, Systolic BP vs. Ten-Year CHD")
    
    fig = px.scatter_3d(data, x="age", y="cigsPerDay", z="sysBP", color="TenYearCHD",
                        color_continuous_scale=["blue", "red"],
                        labels={"age": "Age", "cigsPerDay": "Cigarettes Per Day", "sysBP": "Systolic BP", "TenYearCHD": "CHD"})
    
    fig.update_layout(
        scene=dict(xaxis_title="Age", yaxis_title="Cigarettes Per Day", zaxis_title="Systolic BP"),
        title="Age, Cigarettes Per Day, Systolic BP vs. Ten-Year CHD"
    )
    
    st.plotly_chart(fig)
    
    

#visualization with HiPlot
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file
    
    st.header("Visualization with HiPlot")
    selected_columns = st.multiselect("Select columns to visualize", imputed_data.columns,default = ['age', 'cigsPerDay', 'totChol', 'sysBP','heartRate','TenYearCHD'])
                                      #color='TenYearCHD', title='Interactive Parallel Coordinates Plot')
    selected_data = imputed_data[selected_columns]
    if not selected_data.empty:
        experiment = hip.Experiment.from_dataframe(selected_data)
        hiplot_html_file = save_hiplot_to_html(experiment)
        st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
    else:
        st.write("No data selected. Please choose at least one column to visualize.")










    
    
    
    







    





