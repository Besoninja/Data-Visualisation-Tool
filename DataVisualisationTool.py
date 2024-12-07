import streamlit as st
import pandas as pd

# Filtering function
def filter_data(df):
    st.sidebar.header("Filter Your Data")

    # Store filtered and removed entries
    filtered_df = df.copy()
    removed_df = pd.DataFrame()  # Starts empty

    # Let the user select a column to filter
    column = st.sidebar.selectbox("Select a column to filter", df.columns)
    st.sidebar.write(f"**Data Type:** {df[column].dtype}")  # Show the data type

    # Determine the column's data type
    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric filters: Range slider
        min_val, max_val = int(df[column].min()), int(df[column].max())
        user_min, user_max = st.sidebar.slider(
            f"Filter {column}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        # Apply filter
        removed_df = filtered_df[~((filtered_df[column] >= user_min) & (filtered_df[column] <= user_max))]
        filtered_df = filtered_df[(filtered_df[column] >= user_min) & (filtered_df[column] <= user_max)]

    elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == object:
        # Categorical filters: Multiselect
        unique_vals = df[column].dropna().unique()
        selected_vals = st.sidebar.multiselect(
            f"Filter {column}",
            options=unique_vals,
            default=unique_vals
        )
        # Apply filter
        removed_df = filtered_df[~filtered_df[column].isin(selected_vals)]
        filtered_df = filtered_df[filtered_df[column].isin(selected_vals)]

    elif pd.api.types.is_string_dtype(df[column]):
        # Text filters: Substring search
        text_filter = st.sidebar.text_input(f"Search in {column}")
        if text_filter:
            removed_df = filtered_df[~filtered_df[column].str.contains(text_filter, case=False, na=False)]
            filtered_df = filtered_df[filtered_df[column].str.contains(text_filter, case=False, na=False)]

    # Show Removed Entries in the Main App Area
    if not removed_df.empty:
        st.subheader("Removed Entries")
        st.dataframe(removed_df)

        # Allow users to re-add entries
        st.markdown(f"### Re-add Entries to Filtered Data (Column: {column})")
        re_add_vals = st.multiselect(
            "Select entries to re-add:",
            options=removed_df[column].unique(),
        )

        # Re-add selected entries
        if re_add_vals:
            re_added = removed_df[removed_df[column].isin(re_add_vals)]
            filtered_df = pd.concat([filtered_df, re_added])
            removed_df = removed_df[~removed_df[column].isin(re_add_vals)]

    # Return updated filtered DataFrame
    return filtered_df

# Main function
def main():
    st.title("Data Visualization Tool - 2D Tables")
    
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
            else:
                st.error("Unsupported file format!")
                return
            
            # Show data preview with data types
            st.subheader("Data Preview with Data Types")
            data_preview = df.head().copy()
            data_preview.loc['Data Type'] = df.dtypes
            st.dataframe(data_preview)

            # Apply filters
            filtered_df = filter_data(df)
            
            st.subheader("Filtered Data")
            st.dataframe(filtered_df)

            st.subheader("Basic Statistics (Filtered Data)")
            st.write(filtered_df.describe())

        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV or Excel file to start.")

# Run the app
if __name__ == "__main__":
    main()
