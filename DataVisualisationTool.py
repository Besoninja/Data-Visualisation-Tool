"""
Enhanced Data Visualisation Tool
===============================

This script provides an interactive web application (using Streamlit) to upload, preview, clean,
and create various visualisations from a dataset. Users can also perform basic statistical analysis
and export the transformed data. The application supports multiple file formats (CSV, Excel, JSON,
Parquet, and SQLite) and offers common data cleaning operations and customisation of plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import json
import pyarrow.parquet as pq
import sqlite3
from pathlib import Path

# Constants
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "json", "parquet", "sqlite"]
CACHE_DIR = Path("./cache")
MAX_MEMORY_USAGE = 1024 * 1024 * 1024  # 1GB

def initialise_session_state():
    """
    Initialise session state variables to maintain UI selections, theme, and other preferences
    across multiple interactions during a single user session.

    Parameters:
        None

    Returns:
        None

    Example usage:
        initialise_session_state()
    """
    default_values = {
        'selected_columns': [],
        'selected_visualisation_type': None,
        'numeric_columns': [],
        'categorical_columns': [],
        'datetime_columns': [],
        'theme': 'light',
        'cached_data': None,
        'tutorial_shown': False
    }

    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def display_tutorial():
    """
    Display a first-time user tutorial as an expandable section. Once dismissed, it won't reappear.

    Parameters:
        None

    Returns:
        None

    Example usage:
        display_tutorial()
    """
    if not st.session_state.tutorial_shown:
        with st.expander("📚 Getting Started Guide", expanded=True):
            st.markdown("""
            # Welcome to the Enhanced Data Visualisation Tool!
            
            **How to use this tool:**
            1. **Upload Data**: Upload your data file in the sidebar.
            2. **Explore Tabs**: Use the tabs for various analysis types.
            3. **Customise**: Adjust chart options below each visualisation.
            4. **Export**: Download your visualisations or processed data.
            """)
            if st.button("Got it! Don't show again"):
                st.session_state.tutorial_shown = True

def check_memory_usage(dataframe):
    """
    Check the memory usage of the loaded dataset and warn the user if it exceeds a defined threshold.

    Parameters:
        dataframe (pd.DataFrame): The dataset to check for memory usage.

    Returns:
        bool: True if within acceptable memory limits, False if it's too large.

    Example usage:
        if not check_memory_usage(df):
            # Prompt user to sample the data
    """
    memory_usage = dataframe.memory_usage(deep=True).sum()
    if memory_usage > MAX_MEMORY_USAGE:
        st.warning("⚠️ Large dataset detected. Consider using data sampling to improve performance.")
        return False
    return True

def apply_data_cleaning_options(dataframe):
    """
    Apply user-selected data cleaning operations such as handling missing values, removing outliers,
    and removing duplicate rows. These options are presented in the sidebar.

    Parameters:
        dataframe (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.

    Example usage:
        df = apply_data_cleaning_options(df)
    """
    st.sidebar.subheader("Data Cleaning Options")
    
    # Missing values handling strategy
    missing_value_strategy = st.sidebar.selectbox(
        "Handle missing values",
        ["None", "Drop", "Fill with mean", "Fill with median", "Fill with mode"]
    )
    
    # Apply missing value strategy
    if missing_value_strategy == "Drop":
        dataframe = dataframe.dropna()
    elif missing_value_strategy == "Fill with mean":
        dataframe = dataframe.fillna(dataframe.mean(numeric_only=True))
    elif missing_value_strategy == "Fill with median":
        dataframe = dataframe.fillna(dataframe.median(numeric_only=True))
    elif missing_value_strategy == "Fill with mode":
        dataframe = dataframe.fillna(dataframe.mode().iloc[0])  # mode() returns a DataFrame

    # Remove outliers using z-score thresholding if selected
    if st.sidebar.checkbox("Remove outliers"):
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Calculate z-scores and filter out points beyond z=3 to reduce outliers
            z_scores = np.abs(stats.zscore(dataframe[col].dropna()))
            dataframe = dataframe[(z_scores < 3)]

    # Remove duplicate rows if selected
    if st.sidebar.checkbox("Remove duplicate rows"):
        dataframe = dataframe.drop_duplicates()
    
    return dataframe

def create_visualisation(dataframe, visualisation_type, selected_columns, custom_options=None):
    """
    Create and display a visualisation based on user selections. Also provides a download button
    to save the generated visualisation as a PNG image.

    Parameters:
        dataframe (pd.DataFrame): The dataset to visualise.
        visualisation_type (str): The type of visualisation (e.g. "Correlation Matrix", "Box Plot").
        selected_columns (list): Columns from the dataset used in the visualisation.
        custom_options (dict, optional): Dictionary of user-defined customisation options.
                                         Example keys: 'title', 'color_theme', 'show_legend'.

    Returns:
        None

    Example usage:
        create_visualisation(df, "Histogram", ["column_name"], custom_options)
    """
    if custom_options is None:
        custom_options = {}
    
    # Retrieve customisation settings
    title = custom_options.get('title', '')
    colour_theme = custom_options.get('color_theme', 'viridis')
    show_legend = custom_options.get('show_legend', True)

    try:
        # Different plotting logic depending on chosen visualisation type:
        if visualisation_type == "Correlation Matrix":
            numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(dataframe[col])]
            if len(numeric_cols) > 1:
                correlation_values = dataframe[numeric_cols].corr()
                fig = px.imshow(
                    correlation_values,
                    title=title or "Correlation Matrix",
                    color_continuous_scale=colour_theme
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif visualisation_type == "Box Plot":
            if len(selected_columns) >= 2:
                fig = px.box(
                    dataframe,
                    x=selected_columns[0],
                    y=selected_columns[1],
                    title=title or "Box Plot"
                )
                fig.update_traces(marker_color=px.colors.sequential[colour_theme][5])
                st.plotly_chart(fig, use_container_width=True)
        
        elif visualisation_type == "Violin Plot":
            if len(selected_columns) >= 2:
                fig = px.violin(
                    dataframe,
                    x=selected_columns[0],
                    y=selected_columns[1],
                    title=title or "Violin Plot"
                )
                fig.update_traces(marker_color=px.colors.sequential[colour_theme][5])
                st.plotly_chart(fig, use_container_width=True)
        
        elif visualisation_type == "Histogram":
            if selected_columns:
                fig = px.histogram(
                    dataframe,
                    x=selected_columns[0],
                    title=title or "Histogram"
                )
                fig.update_traces(marker_color=px.colors.sequential[colour_theme][5])
                st.plotly_chart(fig, use_container_width=True)
        
        elif visualisation_type in ["Line Chart", "Area Chart", "Bar Chart", "Scatter Plot"]:
            if len(selected_columns) >= 2:
                # For time series or relationship plots, the first selected column often acts as X-axis
                if visualisation_type == "Line Chart":
                    fig = px.line(
                        dataframe,
                        x=selected_columns[0],
                        y=selected_columns[1:],
                        title=title or "Line Chart"
                    )
                elif visualisation_type == "Area Chart":
                    fig = px.area(
                        dataframe,
                        x=selected_columns[0],
                        y=selected_columns[1:],
                        title=title or "Area Chart"
                    )
                elif visualisation_type == "Bar Chart":
                    fig = px.bar(
                        dataframe,
                        x=selected_columns[0],
                        y=selected_columns[1:],
                        title=title or "Bar Chart"
                    )
                else:  # Scatter Plot
                    fig = px.scatter(
                        dataframe,
                        x=selected_columns[0],
                        y=selected_columns[1],
                        title=title or "Scatter Plot"
                    )
                
                fig.update_traces(marker_color=px.colors.sequential[colour_theme][5])
                fig.update_layout(showlegend=show_legend)
                st.plotly_chart(fig, use_container_width=True)
        
        # Download button for the generated visualisation
        if st.button("Download Visualisation"):
            fig.write_image("visualisation.png")
            with open("visualisation.png", "rb") as file:
                st.download_button(
                    label="Download PNG",
                    data=file,
                    file_name="visualisation.png",
                    mime="image/png"
                )

    except Exception as error:
        st.error(f"Error creating visualisation: {str(error)}")

def perform_basic_statistical_analysis(dataframe, selected_columns):
    """
    Conduct basic statistical tests and display the results. Depending on the data types of the chosen columns,
    different tests will be run. For numeric data, Pearson correlation and t-tests are provided.
    For categorical data, chi-square tests are shown.

    Parameters:
        dataframe (pd.DataFrame): The dataset to analyse.
        selected_columns (list): Columns selected by the user for analysis.

    Returns:
        None

    Example usage:
        perform_basic_statistical_analysis(df, ["age", "height"])
    """
    if len(selected_columns) >= 2:
        st.subheader("Two-Variable Analysis")
        col1, col2 = selected_columns[:2]
        
        # If both columns are numeric, show correlation and t-tests
        if pd.api.types.is_numeric_dtype(dataframe[col1]) and pd.api.types.is_numeric_dtype(dataframe[col2]):
            correlation_value = dataframe[col1].corr(dataframe[col2])
            st.write(f"Pearson Correlation between {col1} and {col2}: {correlation_value:.3f}")
            
            t_stat, p_value = stats.ttest_ind(dataframe[col1].dropna(), dataframe[col2].dropna())
            st.write(f"Independent T-test p-value: {p_value:.3f}")
            
            # Normality test (Shapiro-Wilk)
            st.write("### Normality Test (Shapiro-Wilk)")
            for col in [col1, col2]:
                stat, p_val = stats.shapiro(dataframe[col].dropna())
                st.write(f"{col}: p-value = {p_val:.3f}")
        
        # If one or both columns are categorical, run a chi-square test
        elif pd.api.types.is_categorical_dtype(dataframe[col1]) or pd.api.types.is_categorical_dtype(dataframe[col2]):
            contingency_table = pd.crosstab(dataframe[col1], dataframe[col2])
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-square test p-value: {p_val:.3f}")
            st.write("Contingency Table:")
            st.dataframe(contingency_table)

def main():
    """
    The main entry point for the Streamlit application. It sets up the UI layout, handles file uploads,
    applies data cleaning, and sets up different tabs for visualisation and statistical analysis. Users
    can also export data in various formats.

    Parameters:
        None

    Returns:
        None

    Example usage:
        if __name__ == "__main__":
            main()
    """
    st.set_page_config(page_title="Enhanced Data Visualisation Tool", layout="wide")
    initialise_session_state()
    
    # Sidebar: Theme toggle and documentation links
    with st.sidebar:
        if st.checkbox("Dark Mode", key="dark_mode"):
            st.session_state.theme = 'dark'
        else:
            st.session_state.theme = 'light'
        
        st.markdown("---")
        st.markdown("[📚 Documentation](https://github.com/yourusername/data-vis-tool/wiki)")
        st.markdown("[💡 Report an Issue](https://github.com/yourusername/data-vis-tool/issues)")
    
    st.title("Enhanced Data Visualisation Tool")
    display_tutorial()
    
    # File upload section
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=SUPPORTED_FILE_TYPES)
    
    if uploaded_file is not None:
        try:
            # Load data depending on file extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                selected_sheet = st.sidebar.selectbox("Select a sheet", sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.sqlite'):
                conn = sqlite3.connect(uploaded_file)
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                selected_table = st.sidebar.selectbox("Select a table", tables['name'])
                df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                conn.close()
            
            # Check if data is too large and suggest sampling if needed
            if not check_memory_usage(df):
                sample_size = st.sidebar.slider("Select sample size (%)", 1, 100, 10)
                df = df.sample(frac=sample_size/100)
            
            # Data cleaning
            df = apply_data_cleaning_options(df)
            
            # Data overview section
            with st.expander("Data Overview", expanded=True):
                st.write("Preview of your data:")
                st.dataframe(df.head())
                
                st.subheader("Data Info")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            
            # Update session state with data column types
            st.session_state.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            st.session_state.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            st.session_state.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Setup tabs for different analyses
            tabs = st.tabs([
                "Basic Stats",
                "Time Series",
                "Relationships",
                "Distributions",
                "Statistical Tests"
            ])
            
            # Column selection in the sidebar for visualisation
            with st.sidebar:
                st.session_state.selected_columns = st.multiselect(
                    "Select columns for visualisation",
                    options=df.columns.tolist(),
                    default=st.session_state.numeric_columns[:2] if st.session_state.numeric_columns else []
                )
            
            # Visualisation customisation options
            with st.sidebar:
                st.subheader("Customise Visualisation")
                custom_options = {
                    'title': st.text_input("Chart Title"),
                    'color_theme': st.selectbox("Colour Theme", ['viridis', 'plasma', 'inferno', 'magma']),
                    'show_legend': st.checkbox("Show Legend", value=True)
                }
            
            # Basic Statistics Tab
            with tabs[0]:
                st.header("Basic Statistics")
                if st.session_state.numeric_columns:
                    st.dataframe(df[st.session_state.numeric_columns].describe())
            
            # Time Series Tab
            with tabs[1]:
                st.header("Time Series Analysis")
                time_series_types = ["Line Chart", "Area Chart"]
                selected_time_series_viz = st.selectbox("Select visualisation type", time_series_types, key="time_viz")
                if st.session_state.selected_columns:
                    create_visualisation(df, selected_time_series_viz, st.session_state.selected_columns, custom_options)
            
            # Relationships Tab
            with tabs[2]:
                st.header("Relationship Analysis")
                relationship_types = ["Scatter Plot", "Correlation Matrix", "Box Plot", "Violin Plot"]
                selected_relationship_viz = st.selectbox("Select visualisation type", relationship_types, key="rel_viz")
                if st.session_state.selected_columns:
                    create_visualisation(df, selected_relationship_viz, st.session_state.selected_columns, custom_options)
            
            # Distributions Tab
            with tabs[3]:
                st.header("Distribution Analysis")
                distribution_types = ["Histogram", "Bar Chart"]
                selected_distribution_viz = st.selectbox("Select visualisation type", distribution_types, key="dist_viz")
                if st.session_state.selected_columns:
                    create_visualisation(df, selected_distribution_viz, st.session_state.selected_columns, custom_options)
            
            # Statistical Tests Tab
            with tabs[4]:
                st.header("Statistical Analysis")
                if st.session_state.selected_columns:
                    perform_basic_statistical_analysis(df, st.session_state.selected_columns)
            
            # Export options in the sidebar
            st.sidebar.header("Export Options")
            export_format = st.sidebar.selectbox("Export Format", ["CSV", "Excel", "JSON", "Parquet"])
            
            if st.sidebar.button("Export Data"):
                if export_format == "CSV":
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.sidebar.download_button("Download CSV", csv_data, "data_export.csv", "text/csv")
                elif export_format == "Excel":
                    buffer = io.BytesIO()
                    df.to_excel(buffer, index=False)
                    st.sidebar.download_button("Download Excel", buffer.getvalue(), "data_export.xlsx")
                elif export_format == "JSON":
                    json_str = df.to_json(orient='records')
                    st.sidebar.download_button("Download JSON", json_str, "data_export.json")
                elif export_format == "Parquet":
                    buffer = io.BytesIO()
                    df.to_parquet(buffer)
                    st.sidebar.download_button("Download Parquet", buffer.getvalue(), "data_export.parquet")
        
        except Exception as error:
            st.error(f"Error processing file: {str(error)}")
    else:
        st.info("Please upload a file to start. Supported formats: " + ", ".join(SUPPORTED_FILE_TYPES))

if __name__ == "__main__":
    main()
