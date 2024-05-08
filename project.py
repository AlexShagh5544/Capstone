#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd

# Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns

# for splitting the data into train and test
from sklearn.model_selection import train_test_split

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb

# Evaluation Libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Encoding Library
from sklearn.preprocessing import LabelEncoder

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set the configuration option to disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)



# Cleaning data Function
def clean_data(df, fill_methods, numeric_columns_with_outliers, remove_outliers=False, normalization='none'):
    # Handling missing values for selected columns
    for column, fill_method in fill_methods.items():
        if fill_method == 'forward_fill':
            df[column].fillna(method='ffill', inplace=True)
        elif fill_method == 'backward_fill':
            df[column].fillna(method='bfill', inplace=True)
        elif fill_method == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif fill_method == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif fill_method == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)
    
    # Handling outliers with Z-score method
    if remove_outliers:
        from scipy import stats
        for column in numeric_columns_with_outliers:
            z_scores = np.abs(stats.zscore(df[column]))
            df = df[(z_scores < 3)]
    
    # Normalization or Standardization
    if normalization == 'min_max':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    elif normalization == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    
    return df




# Visualization data Function based on user selections
def generate_visualization(df, chart_type="histogram", selected_columns=None, target_variable=None, colors=None):
    fig, ax = plt.subplots()
    st.subheader(f"{chart_type.capitalize()} Visualization")

    if chart_type == "histogram":
        for column in selected_columns:
            sns.histplot(df[column], ax=ax, kde=True, bins=30, color=colors)
            plt.xlabel(column) 
            plt.ylabel("Count")
            plt.title(f"Histogram of {column} Column")
            st.pyplot(fig)

    elif chart_type == "bar chart":
        if target_variable and selected_columns:
            for column in selected_columns:
                fig, ax = plt.subplots()
                sns.barplot(x=column, y=target_variable, data=df, ax=ax, color=colors)
                plt.xlabel(column) 
                plt.ylabel(target_variable)
                plt.title(f"Bar Chart: {column} vs {target_variable}")
                st.pyplot(fig)

    elif chart_type == "scatter plot":
        if x_column and y_columns:
            for y_column in y_columns:
                fig, ax = plt.subplots()
                # Check if hue_column is selected and not empty
                if hue_column:  
                    # a custom color palette for hue
                    custom_palette = sns.color_palette("husl", len(df[hue_column].unique()))
                    sns.scatterplot(x=df[x_column], y=df[y_column], hue=df[hue_column], ax=ax, palette=custom_palette)
                else:
                    sns.scatterplot(x=df[x_column], y=df[y_column], ax=ax, color=colors)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                if hue_column:
                    plt.legend(title=hue_column)
                plt.title(f"Scatter Plot: {x_column} vs {y_column}")
                st.pyplot(fig)


    elif chart_type == "line chart":
        if target_variable and selected_columns:
            for column in selected_columns:
                fig, ax = plt.subplots()
                sns.lineplot(x=column, y=target_variable, data=df, ax=ax, color=colors)
                plt.xlabel(column)  
                plt.ylabel(target_variable)  
                plt.title(f"Line Chart: {column} vs {target_variable}")
                st.pyplot(fig)
                
    elif chart_type == "boxplot":
        if target_variable and selected_columns:
            for column in selected_columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=column, y=target_variable, data=df, ax=ax)
                plt.xlabel(column)  
                plt.ylabel(target_variable)  
                plt.title(f"Boxplot: {column} vs {target_variable}")
                st.pyplot(fig)
    

    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

    else:
        st.error("Invalid chart type selected.")
        
        
        
        
# Function to select and train the model
def train_selected_model(X_train, y_train, selected_model):
    if selected_model == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    elif selected_model == 'SVM':
        model = svm.SVC(random_state = 42)
    
    elif selected_model == "XGBoost Classifier":
        model = xgb.XGBClassifier()
    else:
        st.write("Selected model is not supported.")
        return None
    model.fit(X_train, y_train)
    return model


# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    return cm, report

        
        
# Streamlit Application 
st.title('Fraud Detection Tool')
st.write('The application is designed for performing data cleaning, visualization, and fraud detection by model training.')

# Add tabs for different functionalities
tab1, tab2, tab3  = st.tabs(["Data Cleaning", "Data Visualization", "Model Development"])

with tab1:
    st.header('Data Cleaning')
    st.write('Upload a dataset for data cleaning')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        
        # Identify columns with missing values
        columns_with_missing = df.columns[df.isnull().any()].tolist()
        if len(columns_with_missing) > 0:
            st.write("Columns with missing values:")
            missing_data = df.isnull().sum()
            total_values = df.shape[0]
            for col in columns_with_missing:
                missing_count = missing_data[col]
                missing_percent = (missing_count / total_values) * 100
                st.write(f"{col}: {missing_count} missing values ({missing_percent:.2f}%)")
            
            # Create a dictionary to store fill methods for each column
            fill_methods = {}
            for column in columns_with_missing:
                fill_methods[column] = st.selectbox(f"Select method to handle missing values for column '{column}':", 
                                                    ['none', 'forward_fill', 'backward_fill', 'mean', 'median', 'mode'])

            # Identify numeric columns for outlier removal
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns_with_outliers = st.multiselect("Select numeric columns for outlier removal:", numeric_columns)
            
            
            # Option for handling outliers
            remove_outliers = st.checkbox("Remove outliers")

            # Options for data normalization/standardization
            normalization = st.selectbox("Select Normalization/Standardization method:", ['none', 'min_max', 'standard'])

            

            # Button to clean data
            if st.button('Clean Data'):
                df_clean = clean_data(df, fill_methods, numeric_columns_with_outliers, remove_outliers, normalization)
                st.write(df_clean)

                # Box plots to visualize data distribution before and after outlier removal
                if remove_outliers:
                    st.subheader("Box Plot before Outlier Removal")
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df[numeric_columns_with_outliers])
                    st.pyplot()

                    st.subheader("Box Plot after Outlier Removal")
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df_clean[numeric_columns_with_outliers])
                    st.pyplot()

                # Convert DataFrame to CSV and allow user to download
                csv = df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download cleaned data as CSV",
                    data=csv,
                    file_name='cleaned_data.csv',
                    mime='text/csv',
                )
        else:
            st.write("No missing values found in the dataset.")
            
        

            
            
            
with tab2:
    with st.container():
        st.header('Data Visualization')
        st.write('Upload a dataset for making visualizations')
        uploaded_file = st.file_uploader("Upload your dataset (.csv)", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            chart_type = st.selectbox("Choose chart type:", ["histogram", "bar chart", "line chart", "scatter plot", "boxplot", "heatmap"])

            selected_columns = []
            target_variable = None

            if chart_type in ["histogram", "bar chart", "line chart","boxplot"]:
                colors = st.color_picker("Choose color for the plot:", "#FF5733")
                selected_columns = st.multiselect('Select columns to visualize:', df.columns)

            if chart_type in ["bar chartbar", "line chart", "boxplot"]:
                target_variable = st.selectbox("Choose y-axis variable:", [''] + list(df.columns))

            if chart_type == "scatter plot":
                colors = st.color_picker("Choose color for the plot:", "#FF5733")
                x_column = st.selectbox("Choose the X-axis variable:", df.columns)
                y_columns = st.multiselect("Choose the Y-axis variable(s):", df.columns, default=df.columns[1])
                hue_column = st.selectbox("Optional: Choose a variable for color coding (hue):", [''] + list(df.columns))
            if chart_type == "heatmap":
                colors =    None

            
            if st.button('Generate Visualizations'):
                generate_visualization(df, chart_type, selected_columns, target_variable, colors)
                
                
                
                
with tab3:
    st.header('Model Development')
    st.write('Upload a dataset for model training')
    uploaded_file = st.file_uploader("Upload your financial dataset", key="model_development")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.iloc[:100000]
        st.write(df.head())
        
        # Selection of features and target variable
        all_columns = df.columns.tolist()
        selected_features = st.multiselect('Select Features', all_columns, default=all_columns[:-1])
        selected_target = st.selectbox('Select Target Variable', all_columns, index=len(all_columns)-1)
        
        # Convert categorical variables if necessary
        st.write("After Encoding:")
        label_encoders = {}
        for column in selected_features:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
        st.write(df)

        
        selected_model = st.selectbox('Select Algorithm', ['Random Forest', 'SVM', 'XGBoost Classifier'])
        
        if st.button('Train Model'):
            X = df[selected_features]
            y = df[selected_target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = train_selected_model(X_train, y_train, selected_model)
            st.session_state.model = model
            st.session_state.model_trained = True
            
            if model is not None:
                st.session_state.model = model
                st.session_state.model_trained = True
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                report = classification_report(y_test, predictions)
                cm = confusion_matrix(y_test, predictions)

                # Model Evaluation 
                st.subheader('Model Evaluation')
                st.write('**Accuracy:** {:.2f}%'.format(accuracy * 100))

                # Classification Report
                st.subheader('Classification Report')
                st.text(report)

                # Confusion matrix
                st.subheader('Confusion Matrix')
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='coolwarm', linewidths=0.5, linecolor='black')
                plt.xlabel('Predicted', fontsize=12)
                plt.ylabel('Actual', fontsize=12)
                plt.title('Confusion Matrix', fontsize=14)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                st.pyplot(fig)
            
        # Predict New Transactions
        st.header('Predict New Transactions')
        uploaded_sample = st.file_uploader("Upload Data to Predict New Transactions", type=".csv")
        if uploaded_sample is not None:
            # Read the sample data using pandas
            sample_df = pd.read_csv(uploaded_sample)
            # Train a model on the sample data using selected features
            sample_X = sample_df[selected_features]
            st.write(sample_X.head())
            
            # Check if the model is trained
            if st.session_state.model_trained == True:
                for column in selected_features:
                    if sample_X[column].dtype == 'object':
                        le = label_encoders[column]
                        sample_X[column] = le.transform(sample_X[column])
                model = st.session_state.model 
                if st.button('Predict New Transactions'):
                    predictions = model.predict(sample_X)
                    # Combine predictions with original data
                    sample_df['predictions'] = predictions
                    st.write(sample_df.head()) 
                    # Save the combined DataFrame to CSV
                    sample_df.to_csv('predictions.csv', index=False)
                    st.success('Predictions saved to predictions.csv')
            
            else:
                st.write("Please train a model first.")

