import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    # Display Removed Entries
    if not removed_df.empty:
        st.sidebar.markdown("### Removed Entries")
        removed_vals = st.sidebar.multiselect(
            f"Re-add removed {column} entries",
            options=removed_df[column].unique(),
        )

        # Move selected entries back to the filtered list
        if removed_vals:
            re_added = removed_df[removed_df[column].isin(removed_vals)]
            filtered_df = pd.concat([filtered_df, re_added])
            removed_df = removed_df[~removed_df[column].isin(removed_vals)]

    # Display filtered entries
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

            # Add the 'Explore' button to navigate to the EDA section
            if st.button("Explore"):
                st.write("## Exploratory Data Analysis")
                
                # Identify numeric and categorical columns
                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Correlation matrix for numeric columns
                if numeric_cols:
                    st.subheader("Correlation Matrix")
                    corr_matrix = filtered_df[numeric_cols].corr()

                    fig, ax = plt.subplots()
                    cax = ax.matshow(corr_matrix, cmap='coolwarm')
                    fig.colorbar(cax)
                    ax.set_xticks(range(len(numeric_cols)))
                    ax.set_yticks(range(len(numeric_cols)))
                    ax.set_xticklabels(numeric_cols, rotation=90)
                    ax.set_yticklabels(numeric_cols)
                    st.pyplot(fig)

                # Distribution plot for a selected numeric column
                if numeric_cols:
                    st.subheader("Distribution of a Numeric Column")
                    selected_num_col = st.selectbox("Select a numeric column", numeric_cols)
                    fig, ax = plt.subplots()
                    ax.hist(filtered_df[selected_num_col].dropna(), bins=30)
                    ax.set_xlabel(selected_num_col)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {selected_num_col}")
                    st.pyplot(fig)

                # Bar chart for a selected categorical column
                if categorical_cols:
                    st.subheader("Categorical Column Analysis")
                    selected_cat_col = st.selectbox("Select a categorical column", categorical_cols)
                    cat_counts = filtered_df[selected_cat_col].value_counts()

                    fig, ax = plt.subplots()
                    cat_counts.plot(kind='bar', ax=ax)
                    ax.set_xlabel(selected_cat_col)
                    ax.set_ylabel("Count")
                    ax.set_title(f"Counts of {selected_cat_col}")
                    st.pyplot(fig)

                # Scatter plot for two numeric columns
                if len(numeric_cols) > 1:
                    st.subheader("Scatter Plot Between Two Numeric Columns")
                    x_col = st.selectbox("Select X axis", numeric_cols)
                    y_col = st.selectbox("Select Y axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    fig, ax = plt.subplots()
                    ax.scatter(filtered_df[x_col], filtered_df[y_col])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"Scatter plot of {x_col} vs {y_col}")
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV or Excel file to start.")

if __name__ == "__main__":
    main()
