import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def filter_data(df):
    st.sidebar.header("Filter Your Data")
    filtered_df = df.copy()
    removed_df = pd.DataFrame()
    
    # Allow selecting multiple columns to filter simultaneously
    columns_to_filter = st.sidebar.multiselect(
        "Select columns to filter",
        options=df.columns,
        default=[]
    )
    
    for column in columns_to_filter:
        st.sidebar.markdown(f"### Filter {column}")
        if pd.api.types.is_numeric_dtype(df[column]):
            min_val, max_val = float(df[column].min()), float(df[column].max())
            user_min, user_max = st.sidebar.slider(
                f"Range for {column}",
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
                f"Select values for {column}", 
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

    return filtered_df

def plot_correlation_matrix(df, selected_cols):
    if len(selected_cols) > 1:
        corr_matrix = df[selected_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        return fig
    return None

def main():
    st.title("Data Visualization Tool - 2D Tables")
    
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Load data with error handling for encoding issues
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    selected_sheet = st.sidebar.selectbox("Select a sheet", sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            
            # Data Overview Section
            with st.expander("Data Overview", expanded=True):
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                st.subheader("Data Info")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            
            # Filter Data
            filtered_df = filter_data(df)
            
            # EDA Section with column selection
            st.write("## Exploratory Data Analysis")
            
            # Allow users to select which columns to analyze
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Column selection for different plot types
            eda_cols = st.multiselect(
                "Select columns for analysis",
                options=filtered_df.columns.tolist(),
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
            )
            
            selected_numeric = [col for col in eda_cols if col in numeric_cols]
            selected_categorical = [col for col in eda_cols if col in categorical_cols]
            
            # Correlation matrix for selected numeric columns
            if selected_numeric:
                st.subheader("Correlation Matrix")
                corr_fig = plot_correlation_matrix(filtered_df, selected_numeric)
                if corr_fig:
                    st.pyplot(corr_fig)
                
                # Distribution plots for selected numeric columns
                st.subheader("Distribution Plots")
                cols_per_row = 2
                for i in range(0, len(selected_numeric), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(selected_numeric):
                            with col:
                                fig, ax = plt.subplots()
                                sns.histplot(filtered_df[selected_numeric[i + j]], kde=True, ax=ax)
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
            
            # Categorical analysis for selected categorical columns
            if selected_categorical:
                st.subheader("Categorical Analysis")
                for cat_col in selected_categorical:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=filtered_df, y=cat_col, order=filtered_df[cat_col].value_counts().index)
                    plt.title(f"Distribution of {cat_col}")
                    st.pyplot(fig)
            
            # Export filtered data
            if st.button("Download Filtered Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        st.info("Please upload a CSV or Excel file to start.")

if __name__ == "__main__":
    main()
