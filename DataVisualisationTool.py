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
import io
from scipy import stats
import sqlite3
from pathlib import Path

# Constants
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "json", "parquet", "sqlite"]
CACHE_DIR = Path("./cache")
MAX_MEMORY_USAGE = 1024 * 1024 * 1024  # 1GB
MAX_SHAPIRO_SIZE = 5000  # Maximum number of samples for Shapiro test to avoid errors on large data

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
        dataframe (pd.DataFrame): The dataset to check.

    Returns:
        bool: True if within acceptable memory limits, False if too large.

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
    
    if missing_value_strategy == "Drop":
        dataframe = dataframe.dropna()
    elif missing_value_strategy == "Fill with mean":
        dataframe = dataframe.fillna(dataframe.mean(numeric_only=True))
    elif missing_value_strategy == "Fill with median":
        dataframe = dataframe.fillna(dataframe.median(numeric_only=True))
    elif missing_value_strategy == "Fill with mode":
        mode_vals = dataframe.mode()
        if not mode_vals.empty:
            dataframe = dataframe.fillna(mode_vals.iloc[0])

    # Remove outliers using z-score thresholding if selected
    if st.sidebar.checkbox("Remove outliers"):
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(dataframe[col].dropna()))
            dataframe = dataframe[(z_scores < 3)]

    # Remove duplicate rows if selected
    if st.sidebar.checkbox("Remove duplicate rows"):
        dataframe = dataframe.drop_duplicates()
    
    if dataframe.empty:
        st.warning("The dataset is now empty after cleaning. Please adjust your cleaning options.")
    
    return dataframe

def is_viz_compatible(dataframe, viz_type, selected_columns):
    """
    Check if the chosen visualisation type is compatible with the selected columns.
    If not compatible, returns False and displays a warning.

    Parameters:
        dataframe (pd.DataFrame): The dataset.
        viz_type (str): Chosen visualisation type.
        selected_columns (list): Columns chosen for the visualisation.

    Returns:
        bool: True if compatible, False otherwise.

    Example usage:
        if is_viz_compatible(df, "Bar Chart", ["col1", "col2"]):
            create_visualisation(df, "Bar Chart", ["col1", "col2"])
    """
    if viz_type in ["Bar Chart", "Scatter Plot"]:
        if len(selected_columns) < 2:
            st.warning(f"For a {viz_type}, please select at least two columns (x and y).")
            return False

    if viz_type == "Correlation Matrix":
        numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(dataframe[col])]
        if len(numeric_cols) < 2:
            st.warning("For a Correlation Matrix, please select at least two numeric columns.")
            return False

    if viz_type == "Box Plot" or viz_type == "Violin Plot":
        if len(selected_columns) < 2:
            st.warning(f"For a {viz_type}, please select at least two columns (x and y).")
            return False

    if viz_type == "Histogram":
        if len(selected_columns) < 1:
            st.warning("For a Histogram, please select at least one column.")
            return False

    return True

def create_visualisation(dataframe, visualisation_type, selected_columns, custom_options=None):
    """
    Create and display a visualisation based on user selections. Also provides a download button
    to save the generated visualisation as a PNG image directly from memory.

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
    
    title = custom_options.get('title', '')
    colour_theme_name = custom_options.get('color_theme', 'viridis')
    show_legend = custom_options.get('show_legend', True)

    # Map the user chosen theme to Plotly sequential colour lists
    theme_map = {
        'viridis': px.colors.sequential.Viridis,
        'plasma': px.colors.sequential.Plasma,
        'inferno': px.colors.sequential.Inferno,
        'magma': px.colors.sequential.Magma
    }
    colors = theme_map.get(colour_theme_name, px.colors.sequential.Viridis)
    chosen_color = colors[min(5, len(colors)-1)]  # safer indexing

    if not is_viz_compatible(dataframe, visualisation_type, selected_columns):
        return

    try:
        fig = None
        if visualisation_type == "Correlation Matrix":
            numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(dataframe[col])]
            corr_values = dataframe[numeric_cols].corr()
            fig = px.imshow(
                corr_values,
                title=title or "Correlation Matrix",
                color_continuous_scale=colour_theme_name
            )
        
        elif visualisation_type == "Box Plot":
            fig = px.box(
                dataframe,
                x=selected_columns[0],
                y=selected_columns[1],
                title=title or "Box Plot"
            )
            fig.update_traces(marker_color=chosen_color)
        
        elif visualisation_type == "Violin Plot":
            fig = px.violin(
                dataframe,
                x=selected_columns[0],
                y=selected_columns[1],
                title=title or "Violin Plot"
            )
            fig.update_traces(marker_color=chosen_color)
        
        elif visualisation_type == "Histogram":
            fig = px.histogram(
                dataframe,
                x=selected_columns[0],
                title=title or "Histogram"
            )
            fig.update_traces(marker_color=chosen_color)
        
        elif visualisation_type in ["Bar Chart", "Scatter Plot"]:
            if visualisation_type == "Bar Chart":
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
            
            fig.update_traces(marker_color=chosen_color)
            fig.update_layout(showlegend=show_legend)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)

            # Download button for the generated visualisation
            if st.button("Download Visualisation"):
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
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
    if len(selected_columns) < 2:
        st.info("Please select at least two columns for statistical analysis.")
        return
    
    col1, col2 = selected_columns[:2]

    if dataframe.empty:
        st.warning("The dataset is empty, cannot perform statistical analysis.")
        return

    # If both columns are numeric
    if pd.api.types.is_numeric_dtype(dataframe[col1]) and pd.api.types.is_numeric_dtype(dataframe[col2]):
        correlation_value = dataframe[col1].corr(dataframe[col2])
        st.write(f"Pearson Correlation between {col1} and {col2}: {correlation_value:.3f}")
        
        # T-test if we have enough data
        if dataframe[col1].dropna().empty or dataframe[col2].dropna().empty:
            st.warning("Insufficient data for t-test after cleaning.")
        else:
            t_stat, p_value = stats.ttest_ind(dataframe[col1].dropna(), dataframe[col2].dropna())
            st.write(f"Independent T-test p-value: {p_value:.3f}")

        # Normality test (Shapiro-Wilk)
        st.write("### Normality Test (Shapiro-Wilk)")
        for col in [col1, col2]:
            col_data = dataframe[col].dropna()
            if len(col_data) > MAX_SHAPIRO_SIZE:
                col_data = col_data.sample(MAX_SHAPIRO_SIZE, random_state=42)
                st.write(f"Sampling {MAX_SHAPIRO_SIZE} points from {col} for Shapiro test.")
            if col_data.empty:
                st.write(f"Not enough data in {col} to test for normality.")
            else:
                try:
                    _, p_val = stats.shapiro(col_data)
                    st.write(f"{col}: p-value = {p_val:.3f}")
                except Exception:
                    st.write(f"Could not perform Shapiro-Wilk test on {col}.")
    
    # If one or both columns are categorical, run a chi-square test
    elif pd.api.types.is_categorical_dtype(dataframe[col1]) or pd.api.types.is_categorical_dtype(dataframe[col2]):
        contingency_table = pd.crosstab(dataframe[col1], dataframe[col2])
        if contingency_table.empty:
            st.warning("Not enough data for chi-square test.")
        else:
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-square test p-value: {p_val:.3f}")
            st.write("Contingency Table:")
            st.dataframe(contingency_table)
    else:
        st.info("Selected columns do not fit numeric or categorical assumptions for these tests.")

def main():
    """
    The main entry point for the Streamlit application. It sets up the UI layout, handles file uploads,
    applies data cleaning, and sets up different tabs for visualisation and statistical analysis. Users
    can also export data in various formats, limited to the user-selected columns.

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
        st.markdown("[📚 Documentation](https://github.com/Besoninja/Data-Visualisation-Tool/wiki)")
        st.markdown("[💡 Report an Issue](https://github.com/Besoninja/Data-Visualisation-Tool/issues)")
    
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
            
            if df.empty:
                st.info("No further analysis can be done as the dataset is empty.")
                return
            
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
            
            # Setup tabs for different analyses (no time series tab)
            tabs = st.tabs([
                "Basic Stats",
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
                else:
                    st.info("No numeric columns available for descriptive statistics.")

            # Relationships Tab
            with tabs[1]:
                st.header("Relationship Analysis")
                relationship_types = ["Scatter Plot", "Correlation Matrix", "Box Plot", "Violin Plot"]
                selected_relationship_viz = st.selectbox("Select visualisation type", relationship_types, key="rel_viz")
                if st.session_state.selected_columns:
                    create_visualisation(df, selected_relationship_viz, st.session_state.selected_columns, custom_options)
                else:
                    st.info("Please select columns in the sidebar to generate a relationship visualisation.")
            
            # Distributions Tab
            with tabs[2]:
                st.header("Distribution Analysis")
                distribution_types = ["Histogram", "Bar Chart"]
                selected_distribution_viz = st.selectbox("Select visualisation type", distribution_types, key="dist_viz")
                if st.session_state.selected_columns:
                    create_visualisation(df, selected_distribution_viz, st.session_state.selected_columns, custom_options)
                else:
                    st.info("Please select columns in the sidebar to generate a distribution visualisation.")
            
            # Statistical Tests Tab
            with tabs[3]:
                st.header("Statistical Analysis")
                if st.session_state.selected_columns:
                    perform_basic_statistical_analysis(df, st.session_state.selected_columns)
                else:
                    st.info("Please select columns in the sidebar to run statistical tests.")
            
            # Export options in the sidebar (only user-selected columns)
            st.sidebar.header("Export Options")
            export_format = st.sidebar.selectbox("Export Format", ["CSV", "Excel", "JSON", "Parquet"])
            if st.sidebar.button("Export Data"):
                export_cols = st.session_state.selected_columns if st.session_state.selected_columns else []
                if not export_cols:
                    st.sidebar.warning("No columns selected. Please select columns first.")
                else:
                    export_df = df[export_cols]
                    if export_format == "CSV":
                        csv_data = export_df.to_csv(index=False).encode('utf-8')
                        st.sidebar.download_button("Download CSV", csv_data, "data_export.csv", "text/csv")
                    elif export_format == "Excel":
                        buffer = io.BytesIO()
                        export_df.to_excel(buffer, index=False)
                        st.sidebar.download_button("Download Excel", buffer.getvalue(), "data_export.xlsx")
                    elif export_format == "JSON":
                        json_str = export_df.to_json(orient='records')
                        st.sidebar.download_button("Download JSON", json_str, "data_export.json")
                    elif export_format == "Parquet":
                        buffer = io.BytesIO()
                        export_df.to_parquet(buffer)
                        st.sidebar.download_button("Download Parquet", buffer.getvalue(), "data_export.parquet")
        
        except Exception as error:
            st.error(f"Error processing file: {str(error)}")
    else:
        st.info("Please upload a file to start. Supported formats: " + ", ".join(SUPPORTED_FILE_TYPES))

if __name__ == "__main__":
    main()
