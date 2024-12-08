import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

#-------------------------------------------
# Filtering function
#-------------------------------------------
def filter_data(df):
    st.sidebar.header("Filter Your Data")

    # Make a copy of the DataFrame to filter
    filtered_df = df.copy()
    removed_df = pd.DataFrame()

    # Allow the user to select a single column to filter at a time
    # If you want multiple filters simultaneously, you'd need a more complex approach.
    column = st.sidebar.selectbox("Select a column to filter", df.columns)
    st.sidebar.write(f"**Data Type:** {df[column].dtype}")

    # Determine the column's data type and apply appropriate filters
    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric filtering using a range slider
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
        # Categorical filtering using a multiselect
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
        # Text-based filtering using substring search
        text_filter = st.sidebar.text_input(f"Search in {column}")
        if text_filter:
            mask = filtered_df[column].str.contains(text_filter, case=False, na=False)
            removed_df = filtered_df[~mask]
            filtered_df = filtered_df[mask]

    # Allow re-adding removed entries if any
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
            removed_df = removed_df[~removed_df[column].isin(re_added_vals)]

    return filtered_df

#-------------------------------------------
# Main application
#-------------------------------------------
def main():
    st.title("Data Visualization Tool - 2D Tables")
    
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    # Use session state to store data after filtering/upload to avoid losing state on reruns
    if uploaded_file is not None:
        # Load the data
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

            # Show data preview
            st.subheader("Data Preview")
            st.write("Below are the first 5 rows of your data:")
            st.dataframe(df.head())

            # Show data types separately
            st.subheader("Data Types")
            dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
            st.dataframe(dtype_df)

            # Filter Data
            filtered_df = filter_data(df)

            st.subheader("Filtered Data")
            st.dataframe(filtered_df)

            # Basic statistics of the filtered data
            # Check if we have numeric data for describe to work properly
            numeric_data = filtered_df.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                st.subheader("Basic Statistics (Filtered Data)")
                st.write(filtered_df.describe())
            else:
                st.subheader("Basic Statistics (Filtered Data)")
                st.write("No numeric columns to describe.")

            # Show or hide the EDA section using a checkbox for persistence
            show_eda = st.checkbox("Show Exploratory Data Analysis")

            if show_eda:
                st.write("## Exploratory Data Analysis")

                # Identify numeric and categorical columns
                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()

                # Correlation matrix for numeric columns
                if numeric_cols and len(numeric_cols) > 1:
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
                else:
                    st.subheader("Correlation Matrix")
                    st.write("Not enough numeric columns for a correlation matrix.")

                # Distribution of a numeric column
                if numeric_cols:
                    st.subheader("Distribution of a Numeric Column")
                    selected_num_col = st.selectbox("Select a numeric column", numeric_cols)
                    fig, ax = plt.subplots()
                    filtered_df[selected_num_col].dropna().hist(bins=30, ax=ax)
                    ax.set_xlabel(selected_num_col)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {selected_num_col}")
                    st.pyplot(fig)
                else:
                    st.subheader("Distribution of a Numeric Column")
                    st.write("No numeric columns available.")

                # Bar chart for a categorical column
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
                else:
                    st.subheader("Categorical Column Analysis")
                    st.write("No categorical columns available.")

                # Scatter plot between two numeric columns
                if numeric_cols and len(numeric_cols) > 1:
                    st.subheader("Scatter Plot Between Two Numeric Columns")
                    x_col = st.selectbox("Select X axis", numeric_cols)
                    # Ensure Y axis defaults to something different from X
                    numeric_cols_y = [col for col in numeric_cols if col != x_col]
                    if numeric_cols_y:
                        y_col = st.selectbox("Select Y axis", numeric_cols_y)
                        fig, ax = plt.subplots()
                        ax.scatter(filtered_df[x_col], filtered_df[y_col])
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"Scatter plot of {x_col} vs {y_col}")
                        st.pyplot(fig)
                    else:
                        st.write("Not enough different numeric columns for a scatter plot.")
                else:
                    st.subheader("Scatter Plot Between Two Numeric Columns")
                    st.write("Not enough numeric columns for a scatter plot.")

        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV or Excel file to start.")


if __name__ == "__main__":
    main()
