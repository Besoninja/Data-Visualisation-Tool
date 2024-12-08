import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io

def initialize_session_state():
    """Initialize session state variables for persistent selections"""
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = []
    if 'selected_viz_type' not in st.session_state:
        st.session_state.selected_viz_type = None
    if 'numeric_cols' not in st.session_state:
        st.session_state.numeric_cols = []
    if 'categorical_cols' not in st.session_state:
        st.session_state.categorical_cols = []

def filter_data(df):
    st.sidebar.header("Filter Your Data")
    filtered_df = df.copy()
    removed_df = pd.DataFrame()

    # Select column to filter
    column = st.sidebar.selectbox("Select a column to filter", df.columns)
    st.sidebar.write(f"**Data Type:** {df[column].dtype}")

    if pd.api.types.is_numeric_dtype(df[column]):
        min_val, max_val = float(df[column].min()), float(df[column].max())
        user_min, user_max = st.sidebar.slider(
            f"Filter {column}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        mask = (filtered_df[column] >= user_min) & (filtered_df[column] <= user_max)
        removed_df = filtered_df[~mask]
        filtered_df = filtered_df[mask]

    elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
        unique_vals = df[column].dropna().unique()
        selected_vals = st.sidebar.multiselect(
            f"Filter {column}", 
            options=unique_vals, 
            default=unique_vals
        )
        mask = filtered_df[column].isin(selected_vals)
        removed_df = filtered_df[~mask]
        filtered_df = filtered_df[mask]

    elif pd.api.types.is_string_dtype(df[column]):
        text_filter = st.sidebar.text_input(f"Search in {column}")
        if text_filter:
            mask = filtered_df[column].str.contains(text_filter, case=False, na=False)
            removed_df = filtered_df[~mask]
            filtered_df = filtered_df[mask]

    # Show removed entries section
    if not removed_df.empty:
        st.sidebar.markdown("### Removed Entries")
        readd_options = removed_df[column].unique()
        re_added_vals = st.sidebar.multiselect(
            f"Re-add removed {column} entries",
            options=readd_options
        )
        if re_added_vals:
            re_added = removed_df[removed_df[column].isin(re_added_vals)]
            filtered_df = pd.concat([filtered_df, re_added], ignore_index=True)

    return filtered_df

def create_visualization(df, viz_type, selected_cols):
    """Create visualization based on type and selected columns"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if viz_type == "Correlation Matrix":
            numeric_cols = [col for col in selected_cols if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) > 1:
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
        
        elif viz_type == "Distribution":
            for col in selected_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    sns.histplot(data=df, x=col, kde=True, ax=ax)
        
        elif viz_type == "Box Plot":
            if len(selected_cols) >= 2:
                sns.boxplot(data=df, x=selected_cols[0], y=selected_cols[1], ax=ax)
            else:
                sns.boxplot(data=df, y=selected_cols[0], ax=ax)
        
        elif viz_type == "Scatter Plot":
            if len(selected_cols) >= 2:
                sns.scatterplot(data=df, x=selected_cols[0], y=selected_cols[1], ax=ax)
        
        elif viz_type == "Line Plot":
            if len(selected_cols) >= 2:
                sns.lineplot(data=df, x=selected_cols[0], y=selected_cols[1], ax=ax)
        
        elif viz_type == "Bar Plot":
            if len(selected_cols) >= 2:
                sns.barplot(data=df, x=selected_cols[0], y=selected_cols[1], ax=ax)
        
        elif viz_type == "Violin Plot":
            if len(selected_cols) >= 2:
                sns.violinplot(data=df, x=selected_cols[0], y=selected_cols[1], ax=ax)
            else:
                sns.violinplot(data=df, y=selected_cols[0], ax=ax)
        
        plt.xticks(rotation=45)
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def main():
    st.title("Data Visualization Tool - 2D Tables")
    initialize_session_state()

    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                selected_sheet = st.sidebar.selectbox("Select a sheet", sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

            # Show data preview
            with st.expander("Data Preview", expanded=True):
                st.write("Below are the first 5 rows of your data:")
                st.dataframe(df.head())
                
                st.subheader("Data Types")
                dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
                st.dataframe(dtype_df)

            # Filter Data
            filtered_df = filter_data(df)

            # Update session state with column types
            st.session_state.numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            st.session_state.categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Create tabs for different visualizations
            viz_tabs = ["Basic Stats", "Distributions", "Relationships", "Comparisons"]
            tabs = st.tabs(viz_tabs)

            # Column selection (persists across tabs)
            if not st.session_state.selected_columns:
                st.session_state.selected_columns = st.session_state.numeric_cols[:2]

            with st.sidebar:
                st.session_state.selected_columns = st.multiselect(
                    "Select columns for visualization",
                    options=filtered_df.columns.tolist(),
                    default=st.session_state.selected_columns
                )

            # Basic Stats Tab
            with tabs[0]:
                st.header("Basic Statistics")
                if st.session_state.numeric_cols:
                    st.write(filtered_df[st.session_state.numeric_cols].describe())

            # Distributions Tab
            with tabs[1]:
                st.header("Distribution Analysis")
                viz_types = ["Distribution", "Box Plot", "Violin Plot"]
                selected_viz = st.selectbox("Select visualization type", viz_types, key="dist_viz")
                if st.session_state.selected_columns:
                    fig = create_visualization(filtered_df, selected_viz, st.session_state.selected_columns)
                    if fig:
                        st.pyplot(fig)

            # Relationships Tab
            with tabs[2]:
                st.header("Relationship Analysis")
                viz_types = ["Scatter Plot", "Line Plot", "Correlation Matrix"]
                selected_viz = st.selectbox("Select visualization type", viz_types, key="rel_viz")
                if st.session_state.selected_columns:
                    fig = create_visualization(filtered_df, selected_viz, st.session_state.selected_columns)
                    if fig:
                        st.pyplot(fig)

            # Comparisons Tab
            with tabs[3]:
                st.header("Comparison Analysis")
                viz_types = ["Bar Plot", "Box Plot", "Violin Plot"]
                selected_viz = st.selectbox("Select visualization type", viz_types, key="comp_viz")
                if st.session_state.selected_columns:
                    fig = create_visualization(filtered_df, selected_viz, st.session_state.selected_columns)
                    if fig:
                        st.pyplot(fig)

            # Export functionality
            if st.button("Download Filtered Data"):
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    "filtered_data.csv",
                    "text/csv",
                    key='download-csv'
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV or Excel file to start.")

if __name__ == "__main__":
    main()
