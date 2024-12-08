import streamlit as st
import pandas as pd
import numpy as np
import io

def initialise_session_state():
    """Initialise session state variables for persistent selections"""
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

def create_visualisation(df, viz_type, selected_cols):
    """Create visualisation based on type and selected columns"""
    try:
        if viz_type == "Correlation Matrix":
            numeric_cols = [col for col in selected_cols if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) > 1:
                corr_data = df[numeric_cols].corr()
                st.dataframe(corr_data.style.background_gradient(cmap='coolwarm'))
        
        elif viz_type == "Line Chart":
            if len(selected_cols) >= 2:
                st.line_chart(df, x=selected_cols[0], y=selected_cols[1:])
        
        elif viz_type == "Area Chart":
            if len(selected_cols) >= 2:
                st.area_chart(df, x=selected_cols[0], y=selected_cols[1:])
        
        elif viz_type == "Bar Chart":
            if len(selected_cols) >= 2:
                st.bar_chart(df, x=selected_cols[0], y=selected_cols[1:])
        
        elif viz_type == "Scatter Plot":
            if len(selected_cols) >= 2:
                st.scatter_chart(df, x=selected_cols[0], y=selected_cols[1:])

    except Exception as e:
        st.error(f"Error creating visualisation: {str(e)}")

def main():
    st.title("Data Visualisation Tool - 2D Tables")
    initialise_session_state()

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
            with st.expander("Data Overview", expanded=True):
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

            # Create tabs for different visualisations
            viz_tabs = ["Basic Stats", "Time Series", "Relationships", "Distributions"]
            tabs = st.tabs(viz_tabs)

            # Column selection (persists across tabs)
            if not st.session_state.selected_columns:
                st.session_state.selected_columns = st.session_state.numeric_cols[:2]

            with st.sidebar:
                st.session_state.selected_columns = st.multiselect(
                    "Select columns for visualisation",
                    options=filtered_df.columns.tolist(),
                    default=st.session_state.selected_columns
                )

            # Basic Stats Tab
            with tabs[0]:
                st.header("Basic Statistics")
                if st.session_state.numeric_cols:
                    st.dataframe(filtered_df[st.session_state.numeric_cols].describe())

            # Time Series Tab
            with tabs[1]:
                st.header("Time Series Analysis")
                viz_types = ["Line Chart", "Area Chart"]
                selected_viz = st.selectbox("Select visualisation type", viz_types, key="time_viz")
                if st.session_state.selected_columns:
                    create_visualisation(filtered_df, selected_viz, st.session_state.selected_columns)

            # Relationships Tab
            with tabs[2]:
                st.header("Relationship Analysis")
                viz_types = ["Scatter Plot", "Correlation Matrix"]
                selected_viz = st.selectbox("Select visualisation type", viz_types, key="rel_viz")
                if st.session_state.selected_columns:
                    create_visualisation(filtered_df, selected_viz, st.session_state.selected_columns)

            # Distributions Tab
            with tabs[3]:
                st.header("Distribution Analysis")
                viz_types = ["Bar Chart"]
                selected_viz = st.selectbox("Select visualisation type", viz_types, key="dist_viz")
                if st.session_state.selected_columns:
                    create_visualisation(filtered_df, selected_viz, st.session_state.selected_columns)

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
