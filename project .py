#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd

# Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns

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

    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

    else:
        st.error("Invalid chart type selected.")

        
        
        
        
# Streamlit Application UI
st.title('Financial Data Analysis App')
st.write('This app allows you to upload a financial dataset and perform data cleaning, visualization, and fraud detection by model training.')

# Add tabs for different functionalities
tab1, tab2 = st.tabs(["Data Cleaning", "Data Visualization"])

with tab1:
    st.header('Data Cleaning')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        
        # Identify columns with missing values
        columns_with_missing = df.columns[df.isnull().any()].tolist()
        if len(columns_with_missing) > 0:
            #st.write("Columns with missing values:", columns_with_missing)
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
            normalization = st.selectbox("Select Normalization/Standardization method:", 
                                        ['none', 'min_max', 'standard'])

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
        uploaded_file = st.file_uploader("Upload your dataset (.csv)", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            chart_type = st.selectbox("Choose chart type:", ["histogram", "bar chart", "line chart", "scatter plot", "heatmap"])

            selected_columns = []
            target_variable = None

            if chart_type in ["histogram", "bar chart", "line chart"]:
                colors = st.color_picker("Choose color for the plot:", "#FF5733")
                selected_columns = st.multiselect('Select columns to visualize:', df.columns)

            if chart_type in ["bar chartbar", "line chart"]:
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

