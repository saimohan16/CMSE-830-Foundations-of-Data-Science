#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.feature_selection import SelectKBest, chi2
import plotly.express as px
import hiplot as hip
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.impute import SimpleImputer
#import pandas as pd
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

# Load the Framingham dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Framingham_app/framingham.csv')  # Replace with the actual path to your dataset
    return data

data = load_data()


tab1, tab2, tab3,tab4,tab5,tab6,tab7 = st.tabs(["About Data", "Basic information Plots", "UniVariant Analysis Plots","MultiVariant Analysis Plots","Machine Learning Models","Inference on Inputs","Bio"])

with tab1:
   st.title("Framingham Heart Study Data")
   image = Image.open('Framingham_app/Heart_img.png')
   img = image.resize((image.height, 300))



   st.image(img, caption='Image')
   
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
    st.subheader("Conclusion")
    st.write(" From the above analysis we can clearly see how the data is distributed and how the missing values look after imputation.")
    

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


with tab5:
    st.title("Machine Learning Classifier Performance")
    #data, selected_columns = oad_data()
    st.write('The Predictive Analysis feature within the application utilizes sophisticated machine learning models such as Logistic Regression and Random Forest to unravel the features of the subjects. With a diverse array of models, users can gain varied analytical perspectives on Framingham Data.')
    st.write('This tab serves as a conduit for translating complex data into accessible and interactive insights, enabling users, from decision-makers to the general public, to experiment with data and witness immediate results. This approach not only facilitates the prediction and comprehension of this complex data but also empowers users to engage with and respond to these critical issues proactively. The customization feature allows users to tailor the analysis with respect to the selected feautures and also use top 10 features to dervide insights.')
    st.write('The inclusion of a range of models is pivotal, as it enables users to apply diverse analytical perspectives to the same dataset. This diversity is critical because different models can spotlight distinct facets of the data.')
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    def oad_data():
        data = pd.read_csv('Framingham_app/framingham.csv')
        
        data = data.sample(frac=0.6, random_state=42)  # Use a fixed random state for reproducibility

        # Preprocessing steps (handle missing values, etc.)
        missing_values_before = data.isnull().sum()
        #st.write("Missing values before imputation:", missing_values_before)

        # Impute missing values for numerical columns
        num_cols = data.select_dtypes(include=['int64', 'float64']).columns
        imputer = SimpleImputer(strategy='median')
        data[num_cols] = imputer.fit_transform(data[num_cols])

        # Check for missing values after imputation
        missing_values_after = data.isnull().sum()

        #st.write("Missing values after imputation:", missing_values_after)

        # Check for and replace infinite values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check for missing values after replacing infinite values
        missing_infinite = data.isnull().sum()
        #st.write("Missing/infinite values after replacement:", missing_infinite)

        # Drop any rows that still have NaNs (should be very few if any)
        data.dropna(inplace=True)

        # Preprocessing steps (handle missing values, etc.)
        # ...

        # Splitting the dataset based on class
        target1 = data[data['TenYearCHD'] == 1]
        target0 = data[data['TenYearCHD'] == 0]

        # Resampling to balance the dataset
        target1_resampled = resample(target1, replace=True, n_samples=len(target0), random_state=40)
        data_balanced = pd.concat([target0, target1_resampled])

        # Feature Selection
        X = data_balanced.iloc[:, 0:15]
        y = data_balanced.iloc[:, -1]

        best = SelectKBest(score_func=chi2, k=10)
        best.fit(X, y)

        # Select the top 10 features
        top_features = [X.columns[i] for i in best.get_support(indices=True)]
        data_selected = data_balanced[top_features + ['TenYearCHD']]

        return data_selected

    data = oad_data()
    
    def train_evaluate_model_cv(model, X, y, cv_folds):
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='f1')
        mean_score = np.mean(scores)
        accuracy_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        mean_accuracy_score = np.mean(accuracy_scores)
        st.write(f"Mean Accuracy Score (Cross-Validation): {mean_accuracy_score:.4f}")
        st.write(f"Mean F1 Score (Cross-Validation): {mean_score:.4f}")
        return mean_score, mean_accuracy_score
        #st.write(f"Mean F1 Score (Cross-Validation): {mean_score:.4f}")
        
        #return mean_score
    # Function to plot top 10 important features
    def plot_top_features(X, y):
        selector = SelectKBest(f_classif, k=10)
        X_new = selector.fit_transform(X, y)
        feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
        top_features = feature_scores.nlargest(10, 'Score')

        fig, ax = plt.subplots()
        top_features.plot(x='Feature', y='Score', kind='barh', ax=ax, color='skyblue')
        ax.set_title('Top 10 Important Features')
        st.pyplot(fig)

    # Main app
    #st.title("Machine Learning Classifier Performance")
    
    #st.title("Machine Learning Classifier Performance")

# Load data
    data = oad_data()

    # Sidebar for feature selection
    all_features = data.drop('TenYearCHD', axis=1).columns.tolist()
    selected_features = st.multiselect("Select Features for Training", all_features, default=all_features[:10])

    # Option to use top 10 features from SelectKBest
    use_top_features = st.checkbox("Use Top 10 Features from SelectKBest", value=True)
    
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    # Prepare data based on feature selection
    if use_top_features:
        X = data[selected_features]
    else:
        X = data[all_features]
    y = data['TenYearCHD']

    # Load data
    data = oad_data()

    # Sidebar for CV folds and feature selection
    cv_folds = st.slider("Select Number of Cross-Validation Folds", min_value=2, max_value=10, value=5)
    #selected_features = st.sidebar.multiselect("Select Features for Training", 
                                           #options=data.columns.drop('TenYearCHD').tolist(), 
                                           #default=data.columns.drop('TenYearCHD').tolist()[:10])

    #selected_features = st.sidebar.multiselect("Select Features for Training", data.columns.drop('TenYearCHD'), default=data.columns.drop('TenYearCHD'))
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    # Prepare data
    #X = data[selected_features]
    #y = data['TenYearCHD']

    # Create tabs for each model
    tab_lr, tab_rf, tab_nb, tab_gb = st.tabs(["Logistic Regression", "Random Forest", "Naive Bayes", "Gradient Boosting"])

    with tab_lr:
        C_lr = st.number_input("C (Regularization parameter)", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
        model_lr = LogisticRegression(C=C_lr, max_iter=1000)
        if st.button('Run Logistic Regression'):
            train_evaluate_model_cv(model_lr, X, y, cv_folds)

    with tab_rf:
        n_estimators_rf = st.slider("Number of trees in the forest", min_value=10, max_value=200, value=100)
        model_rf = RandomForestClassifier(n_estimators=n_estimators_rf)
        if st.button('Run Random Forest'):
            train_evaluate_model_cv(model_rf, X, y, cv_folds)

   # with tab_svm:
   #     C_svm = st.number_input("SVM - C (Regularization parameter)", 0.01, 10.0, step=0.01, value=1.0)
   #     kernel_svm = st.selectbox("SVM - Kernel", ("linear", "rbf", "poly"))
   #     model_svm = SVC(C=C_svm, kernel=kernel_svm, probability=True)
   #     if st.button('Run SVM'):
   #         train_evaluate_model_cv(model_svm, X, y, cv_folds)

    with tab_gb:
        n_estimators_gb = st.slider("Number of boosting stages", min_value=10, max_value=200, value=100)
        learning_rate_gb = st.number_input("Gradient Boosting - Learning Rate", 0.01, 1.0, step=0.01, value=0.1)
        model_gb = GradientBoostingClassifier(n_estimators=n_estimators_gb, learning_rate=learning_rate_gb)
        if st.button('Run Gradient Boosting'):
            train_evaluate_model_cv(model_gb, X, y, cv_folds)
            
    with tab_nb:
    #st.subheader("Naive Bayes Classifier")

    	model_nb = GaussianNB()

    	if st.button('Run Naive Bayes'):
        	train_evaluate_model_cv(model_nb, X, y, cv_folds)

    st.subheader("How each model understands data ?")
    st.write('Logistic Regression: Overview: Logistic Regression is a statistical method used for binary classification problems. It predicts the probability of an instance belonging to a particular class. Application: In the context of Framingham data, Logistic Regression can be used to predict the likelihood of a participant developing a cardiovascular condition based on various input features such as age, cholesterol levels, blood pressure, etc. However becaue of the complex features presnt in the dataset. This models fails to provide accurate predictions')
    st.write('Random Forest: Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions. It is versatile and can be used for both classification and regression tasks. Application: In the Framingham dataset, Random Forest could be employed to identify important features contributing to cardiovascular risk and provide robust predictions by aggregating outputs from multiple decision trees. This models performs the best and it is able to provide predictions irrespective of the features')
    st.write('Naive Bayes: Naive Bayes is a probabilistic classification algorithm based on Bayes theorem. Despite its simplicity, it often performs well, especially in text classification and simple datasets. Application: Naive Bayes can be applied to predict cardiovascular risk in the Framingham dataset by assuming independence between features, making it suitable for situations where this assumption is reasonable.')
    st.write('Gradient Boosting:Gradient Boosting is an ensemble technique that builds a series of weak learners (usually decision trees) sequentially, with each one correcting errors of its predecessor. Application: In the context of the Framingham data, Gradient Boosting can effectively capture complex relationships between features and the target variable, providing accurate predictions by combining the strengths of multiple weak models.')
    st.write('Experimenting with these models on the Framingham dataset allows for a nuanced understanding of their effectiveness in forecasting and dissecting cardiovascular risk. The choice of model depends on the specific characteristics of the data and the complexity of the scenarios being analyzed. By leveraging this diverse set of models, researchers and analysts can tailor their approach to gain comprehensive insights into the multifaceted nature of cardiovascular health and risk prediction.')
    # Plot top features
    plot_top_features(X, y)
    #st.subheader("How each model understands data ?")
    

with tab6:


    # Function to load and preprocess data
    def oad_data():
        data = pd.read_csv('Framingham_app/framingham.csv')
        data = data.sample(frac=0.6, random_state=42)

        # Preprocessing steps
        num_cols = data.select_dtypes(include=['int64', 'float64']).columns
        imputer = SimpleImputer(strategy='median')
        data[num_cols] = imputer.fit_transform(data[num_cols])

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        # Resampling to balance the dataset
        target1 = data[data['TenYearCHD'] == 1]
        target0 = data[data['TenYearCHD'] == 0]
        target1_resampled = resample(target1, replace=True, n_samples=len(target0), random_state=40)
        data_balanced = pd.concat([target0, target1_resampled])

        # Feature Selection
        X = data_balanced.iloc[:, :-1]
        y = data_balanced['TenYearCHD']
        best = SelectKBest(score_func=chi2, k=10)
        best.fit(X, y)
        top_features = [X.columns[i] for i in best.get_support(indices=True)]
        data_selected = data_balanced[top_features + ['TenYearCHD']]

        return data_selected, top_features

    data, top_features = oad_data()

    # Function to train the models
    def train_models(X_train, y_train):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            models[name] = model
        return models

    # Main app
    st.title("Machine Learning Classifier Performance")

    # Split the data
    X = data[top_features]
    y = data['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # User input for prediction
    with st.container():
        st.subheader("Make Predictions")
        columns = ['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke', 'prevalentHyp','diabetes','totChol','sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

        input_data = {}
        for col in top_features:
            if col in ['male', 'currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes','BPMeds']:
                # Binary columns
                input_data[col] = st.selectbox(f"{col.capitalize()}", [0, 1],key=col)
            else:
                # Numerical columns
                # You might need to adjust min_value, max_value, and value based on the actual range of your data
                input_data[col] = st.number_input(f"{col.capitalize()}", min_value=0, max_value=100, value=50,key=col)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
        #input_data = {feature: st.number_input(f"{feature}", value=np.mean(X[feature])) for feature in top_features}
        #if st.button("Predict"):
            #input_df = pd.DataFrame([input_data])
            models = train_models(X_train, y_train)
            predictions = {name: model.predict(input_df)[0] for name, model in models.items()}

        # Displaying the predictions
            for model_name, prediction in predictions.items():
            	if model_name == 'Random Forest':
                	result = "Positive for CHD" if prediction == 1 else "Negative for CHD"
                	st.write(f"{model_name} Prediction: {result}")
            #models = train_models(X_train, y_train)
            #predictions = {name: model.predict(input_df)[0] for name, model in models.items()}
            #for model_name, prediction in predictions.items():
                #st.write(f"{model_name} Prediction: {prediction}")
                
        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                
        st.subheader("Conclusion")
        st.write("Our innovative web application, built upon the rich Framingham dataset, offers users a powerful and insightful platform for comprehensive data analysis and prediction in cardiovascular health.")
        st.write(" The univariate analysis component provides a meticulous examination of individual variables within the Framingham dataset. Users can delve into the distributions, central tendencies, and variations of key parameters, establishing a foundational understanding of each variable's characteristics. Advancing from univariate analysis, our app seamlessly integrates multivariate analysis capabilities, enabling users to uncover complex relationships and dependencies between various cardiovascular risk factors. This sophisticated exploration facilitates a holistic perspective, empowering users to discern intricate patterns and connections that contribute to a comprehensive understanding of cardiovascular health.")
        st.write("The true value of our application lies in its predictive prowess, driven by machine learning classifiers trained on the Framingham dataset. These models offer users the ability to anticipate potential cardiovascular events, assess risk factors, and make informed decisions for proactive health management. The predictive insights gleaned from our models contribute to a more personalized and preventive approach to cardiovascular care. Our user-friendly interface ensures accessibility for a diverse audience, from healthcare professionals to individuals keen on monitoring their cardiovascular health. By seamlessly integrating analytical tools and machine learning models, our app becomes an indispensable resource for deriving actionable insights from the Framingham dataset, ultimately contributing to enhanced cardiovascular risk assessment and personalized health strategies.")
        st.write("In summary, our web application on the Framingham dataset stands as a comprehensive solution, providing a deep dive into data analysis and predictive modeling specific to cardiovascular health. It is poised to make a meaningful impact on healthcare decision-making and individual well-being, aligning with the broader goals of proactive and personalized healthcare management.")

    # Optional: Add model performance metrics or other analyses here
    
with tab7:
    st.title("About the Developer ")

    col1, col2 = st.columns(2)

    col1.subheader("Sai Mohan Gajapaka")
    col1.text("Master's in Data Science, MSU")
    col1.write("As a dedicated Python programmer with a robust foundation in mathematical modeling and deep neural networks, I am currently advancing my journey in data science. My academic and research experiences have nurtured a strong proficiency in statistical analysis and machine learning, fueling my drive to tackle challenging problems with innovative solutions. My passion lies in applying my skills to real-world issues, particularly those that can make a positive social impact. I am constantly seeking opportunities that challenge me to grow and refine my abilities in this dynamic field. Beyond my academic pursuits, I have a keen interest in watching anime, which not only serves as a creative outlet but also often inspires my approach to complex problems. I'm deeply curious about the potential of deep learning and its applications, and I am committed to exploring its frontiers to contribute meaningfully to the field of data science.")
    
    try :
        col2.image("Framingham_app/profile.png")
    except:
     pass


