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

def initialize_session_state():
    """Initialize session state variables for persistent selections and theme"""
    defaults = {
        'selected_columns': [],
        'selected_viz_type': None,
        'numeric_cols': [],
        'categorical_cols': [],
        'datetime_cols': [],
        'theme': 'light',
        'cached_data': None,
        'tutorial_shown': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def show_tutorial():
    """Display first-time user tutorial"""
    if not st.session_state.tutorial_shown:
        with st.expander("📚 Getting Started Guide", expanded=True):
            st.markdown("""
            # Welcome to the Enhanced Data Visualization Tool!
            
            1. **Upload Data**: Start by uploading your data file in the sidebar
            2. **Explore Tabs**: Use different tabs for various analysis types
            3. **Customize**: Modify charts using the options below each visualization
            4. **Export**: Download your visualizations or processed data
            
            Need help? Check the documentation link in the sidebar!
            """)
            if st.button("Got it! Don't show again"):
                st.session_state.tutorial_shown = True

def check_memory_usage(df):
    """Monitor memory usage and warn if approaching limits"""
    memory_usage = df.memory_usage(deep=True).sum()
    if memory_usage > MAX_MEMORY_USAGE:
        st.warning("⚠️ Large dataset detected. Consider using data sampling to improve performance.")
        return False
    return True

def clean_data(df):
    """Provide data cleaning options"""
    st.sidebar.subheader("Data Cleaning Options")
    
    # Handle missing values
    missing_strategy = st.sidebar.selectbox(
        "Handle missing values",
        ["None", "Drop", "Fill with mean", "Fill with median", "Fill with mode"]
    )
    
    if missing_strategy != "None":
        if missing_strategy == "Drop":
            df = df.dropna()
        elif missing_strategy == "Fill with mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == "Fill with median":
            df = df.fillna(df.median(numeric_only=True))
        elif missing_strategy == "Fill with mode":
            df = df.fillna(df.mode().iloc[0])
    
    # Handle outliers
    if st.sidebar.checkbox("Remove outliers"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df = df[(z_scores < 3)]
    
    # Remove duplicates
    if st.sidebar.checkbox("Remove duplicate rows"):
        df = df.drop_duplicates()
    
    return df

def create_advanced_visualization(df, viz_type, selected_cols, customize_options=None):
    """Create enhanced visualizations with customization options"""
    try:
        if customize_options is None:
            customize_options = {}
        
        # Get default customization options
        title = customize_options.get('title', '')
        color_theme = customize_options.get('color_theme', 'viridis')
        show_legend = customize_options.get('show_legend', True)
        
        if viz_type == "Correlation Matrix":
            numeric_cols = [col for col in selected_cols if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) > 1:
                corr_data = df[numeric_cols].corr()
                fig = px.imshow(corr_data,
                              title=title or "Correlation Matrix",
                              color_continuous_scale=color_theme)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            if len(selected_cols) >= 2:
                fig = px.box(df, x=selected_cols[0], y=selected_cols[1],
                           title=title or "Box Plot",
                           color=selected_cols[0] if len(selected_cols) > 2 else None)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Violin Plot":
            if len(selected_cols) >= 2:
                fig = px.violin(df, x=selected_cols[0], y=selected_cols[1],
                              title=title or "Violin Plot",
                              color=selected_cols[0] if len(selected_cols) > 2 else None)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            if selected_cols:
                fig = px.histogram(df, x=selected_cols[0],
                                 title=title or "Histogram",
                                 color=selected_cols[1] if len(selected_cols) > 1 else None)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type in ["Line Chart", "Area Chart", "Bar Chart", "Scatter Plot"]:
            if len(selected_cols) >= 2:
                if viz_type == "Line Chart":
                    fig = px.line(df, x=selected_cols[0], y=selected_cols[1:],
                                title=title or "Line Chart")
                elif viz_type == "Area Chart":
                    fig = px.area(df, x=selected_cols[0], y=selected_cols[1:],
                                title=title or "Area Chart")
                elif viz_type == "Bar Chart":
                    fig = px.bar(df, x=selected_cols[0], y=selected_cols[1:],
                                title=title or "Bar Chart")
                else:  # Scatter Plot
                    fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1],
                                   title=title or "Scatter Plot",
                                   color=selected_cols[2] if len(selected_cols) > 2 else None)
                
                fig.update_layout(showlegend=show_legend)
                st.plotly_chart(fig, use_container_width=True)
        
        # Add download button for the visualization
        if st.button("Download Visualization"):
            fig.write_image("visualization.png")
            with open("visualization.png", "rb") as file:
                btn = st.download_button(
                    label="Download PNG",
                    data=file,
                    file_name="visualization.png",
                    mime="image/png"
                )
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def perform_statistical_analysis(df, selected_cols):
    """Perform basic statistical analysis"""
    if len(selected_cols) >= 2:
        col1, col2 = selected_cols[:2]
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            # Correlation analysis
            correlation = df[col1].corr(df[col2])
            st.write(f"Correlation coefficient between {col1} and {col2}: {correlation:.3f}")
            
            # T-test
            t_stat, p_value = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
            st.write(f"T-test p-value: {p_value:.3f}")

def main():
    st.set_page_config(page_title="Enhanced Data Visualization Tool", layout="wide")
    initialize_session_state()
    
    # Theme toggle
    with st.sidebar:
        if st.checkbox("Dark Mode", key="dark_mode"):
            st.session_state.theme = 'dark'
        else:
            st.session_state.theme = 'light'
    
    st.title("Enhanced Data Visualization Tool")
    show_tutorial()
    
    # File upload with extended support
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=SUPPORTED_FILE_TYPES)
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
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
            
            # Check memory usage
            if not check_memory_usage(df):
                sample_size = st.sidebar.slider("Select sample size (%)", 1, 100, 10)
                df = df.sample(frac=sample_size/100)
            
            # Clean data
            df = clean_data(df)
            
            # Data overview
            with st.expander("Data Overview", expanded=True):
                st.write("Preview of your data:")
                st.dataframe(df.head())
                
                st.subheader("Data Info")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            
            # Update session state
            st.session_state.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.session_state.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            st.session_state.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Create tabs
            tabs = st.tabs([
                "Basic Stats",
                "Time Series",
                "Relationships",
                "Distributions",
                "Statistical Tests"
            ])
            
            # Column selection
            with st.sidebar:
                st.session_state.selected_columns = st.multiselect(
                    "Select columns for visualization",
                    options=df.columns.tolist(),
                    default=st.session_state.numeric_cols[:2] if st.session_state.numeric_cols else []
                )
            
            # Visualization customization
            with st.sidebar:
                st.subheader("Customize Visualization")
                customize_options = {
                    'title': st.text_input("Chart Title"),
                    'color_theme': st.selectbox("Color Theme", ['viridis', 'plasma', 'inferno', 'magma']),
                    'show_legend': st.checkbox("Show Legend", value=True)
                }
            
            # Basic Stats Tab
            with tabs[0]:
                st.header("Basic Statistics")
                if st.session_state.numeric_cols:
                    st.dataframe(df[st.session_state.numeric_cols].describe())
            
            # Time Series Tab
            with tabs[1]:
                st.header("Time Series Analysis")
                viz_types = ["Line Chart", "Area Chart"]
                selected_viz = st.selectbox("Select visualization type", viz_types, key="time_viz")
                if st.session_state.selected_columns:
                    create_advanced_visualization(df, selected_viz, st.session_state.selected_columns, customize_options)
            
            # Relationships Tab
            with tabs[2]:
                st.header("Relationship Analysis")
                viz_types = ["Scatter Plot", "Correlation Matrix", "Box Plot", "Violin Plot"]
                selected_viz = st.selectbox("Select visualization type", viz_types, key="rel_viz")
                if st.session_state.selected_columns:
                    create_advanced_visualization(df, selected_viz, st.session_state.selected_columns, customize_options)
            
            # Distributions Tab
            with tabs[3]:
                st.header("Distribution Analysis")
                viz_types = ["Histogram", "Bar Chart"]
                selected_viz = st.selectbox("Select visualization type", viz_types, key="dist_viz")
                if st.session_state.selected_columns:
                    create_advanced_visualization(df, selected_viz, st.session_state.selected_columns, customize_options)
            
            # Statistical Tests Tab
            with tabs[4]:
                st.header("Statistical Analysis")
                if st.session_state.selected_columns:
                    perform_statistical_analysis(df, st.session_state.selected_columns)
            
            # Export functionality
            st.sidebar.header("Export Options")
            export_format = st.sidebar.selectbox("Export Format", ["CSV", "Excel", "JSON", "Parquet"])
            
            if st.sidebar.button("Export Data"):
                if export_format == "CSV":
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.sidebar.download_button("Download CSV", csv, "data_export.csv", "text/csv")
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
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a file to start. Supported formats: " + ", ".join(SUPPORTED_FILE_TYPES))

if __name__ == "__main__":
    main()
