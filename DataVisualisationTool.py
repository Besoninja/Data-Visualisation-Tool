import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(
    page_title="Data Visualization Tool - 2D Tables",
    layout="wide",
)

# Caching the data loading function to improve performance
@st.cache_data
def load_data(uploaded_file):
    """Load CSV or Excel file into a pandas DataFrame."""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.sidebar.selectbox("Select a sheet", sheet_names)
        return pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    else:
        st.error("Unsupported file format!")
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if 'show_eda' not in st.session_state:
        st.session_state['show_eda'] = False
    if 'filtered_df' not in st.session_state:
        st.session_state['filtered_df'] = pd.DataFrame()
    if 'removed_df' not in st.session_state:
        st.session_state['removed_df'] = pd.DataFrame()

def filter_data(df):
    """Apply user-defined filters to the DataFrame."""
    st.sidebar.header("Filter Your Data")

    # Let the user select a column to filter
    column = st.sidebar.selectbox("Select a column to filter", df.columns)
    st.sidebar.write(f"**Data Type:** {df[column].dtype}")

    # Initialize or reset filtered and removed dataframes
    if 'filtered_df' not in st.session_state or st.session_state['filtered_df'].empty:
        st.session_state['filtered_df'] = df.copy()
        st.session_state['removed_df'] = pd.DataFrame()

    filtered_df = st.session_state['filtered_df']
    removed_df = st.session_state['removed_df']

    # Determine the column's data type and apply appropriate filter
    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric filters: Range slider
        min_val, max_val = float(df[column].min()), float(df[column].max())
        user_min, user_max = st.sidebar.slider(
            f"Filter {column}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        # Apply filter
        condition = (filtered_df[column] >= user_min) & (filtered_df[column] <= user_max)
        removed = filtered_df[~condition]
        filtered = filtered_df[condition]

    elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == object:
        # Categorical filters: Multiselect
        unique_vals = df[column].dropna().unique()
        selected_vals = st.sidebar.multiselect(
            f"Filter {column}",
            options=unique_vals,
            default=list(unique_vals)
        )
        # Apply filter
        condition = filtered_df[column].isin(selected_vals)
        removed = filtered_df[~condition]
        filtered = filtered_df[condition]

    else:
        st.sidebar.warning(f"Unsupported data type for filtering: {df[column].dtype}")
        return st.session_state['filtered_df']

    # Update session state with filtered and removed dataframes
    st.session_state['filtered_df'] = filtered
    st.session_state['removed_df'] = pd.concat([removed_df, removed]).drop_duplicates()

    # Display Removed Entries and allow re-adding them
    if not st.session_state['removed_df'].empty:
        st.sidebar.markdown("### Removed Entries")
        removed_vals = st.sidebar.multiselect(
            f"Re-add removed {column} entries",
            options=st.session_state['removed_df'][column].unique(),
        )

        if removed_vals:
            re_added = st.session_state['removed_df'][st.session_state['removed_df'][column].isin(removed_vals)]
            st.session_state['filtered_df'] = pd.concat([st.session_state['filtered_df'], re_added]).drop_duplicates()
            st.session_state['removed_df'] = st.session_state['removed_df'][~st.session_state['removed_df'][column].isin(removed_vals)]

    return st.session_state['filtered_df']

def display_data_preview(df):
    """Display a preview of the data with data types."""
    st.subheader("Data Preview with Data Types")
    data_preview = df.head().copy()
    data_preview.loc['Data Type'] = df.dtypes
    st.dataframe(data_preview.style.highlight_max(axis=0))

def display_filtered_data(df):
    """Display the filtered DataFrame."""
    st.subheader("Filtered Data")
    st.dataframe(df)

def display_basic_statistics(df):
    """Display basic statistics of the filtered data."""
    st.subheader("Basic Statistics (Filtered Data)")
    st.write(df.describe())

def display_eda_section(df):
    """Display the Exploratory Data Analysis (EDA) section."""
    st.write("## Exploratory Data Analysis")

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Correlation matrix for numeric columns
    if numeric_cols:
        st.subheader("Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(corr_matrix, cmap='coolwarm')
        fig.colorbar(cax)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=90)
        ax.set_yticklabels(numeric_cols)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation matrix.")

    # Distribution plot for a selected numeric column
    if numeric_cols:
        st.subheader("Distribution of a Numeric Column")
        selected_num_col = st.selectbox("Select a numeric column for distribution", numeric_cols, key='dist_num_col')
        fig, ax = plt.subplots()
        ax.hist(df[selected_num_col].dropna(), bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel(selected_num_col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {selected_num_col}")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for distribution plots.")

    # Bar chart for a selected categorical column
    if categorical_cols:
        st.subheader("Categorical Column Analysis")
        selected_cat_col = st.selectbox("Select a categorical column for bar chart", categorical_cols, key='bar_cat_col')
        cat_counts = df[selected_cat_col].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        cat_counts.plot(kind='bar', ax=ax, color='coral')
        ax.set_xlabel(selected_cat_col)
        ax.set_ylabel("Count")
        ax.set_title(f"Counts of {selected_cat_col}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No categorical columns available for bar charts.")

    # Scatter plot for two numeric columns
    if len(numeric_cols) > 1:
        st.subheader("Scatter Plot Between Two Numeric Columns")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X axis", numeric_cols, key='scatter_x_col')
        with col2:
            y_col = st.selectbox("Select Y axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key='scatter_y_col')

        if x_col and y_col:
            fig, ax = plt.subplots()
            ax.scatter(df[x_col], df[y_col], alpha=0.7, edgecolors='w', linewidth=0.5)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns available for scatter plots.")

def main():
    """Main function to run the Streamlit app."""
    st.title("📊 Data Visualization Tool - 2D Tables")

    # Initialize session state variables
    initialize_session_state()

    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None and not df.empty:
            # Display data preview
            display_data_preview(df)

            # Apply filters
            filtered_df = filter_data(df)

            # Display filtered data and statistics
            display_filtered_data(filtered_df)
            display_basic_statistics(filtered_df)

            # Explore button to show/hide EDA section
            if st.button("🔍 Explore"):
                st.session_state['show_eda'] = True

            # Display EDA section if enabled
            if st.session_state.get('show_eda', False):
                display_eda_section(filtered_df)

        else:
            st.error("The uploaded file is empty or could not be processed.")
    else:
        st.info("🗂️ Please upload a CSV or Excel file to start.")

if __name__ == "__main__":
    main()
