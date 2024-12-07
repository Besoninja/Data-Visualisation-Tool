import streamlit as st
import pandas as pd

def main():
    st.title("Data Visualization Tool - 2D Tables")
    
    # Sidebar for file upload
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    # If a file is uploaded
    if uploaded_file is not None:
        try:
            # Handle CSV or Excel files
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format!")
                return
            
            # Display the uploaded data
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Display basic statistics
            st.subheader("Basic Statistics")
            st.write(df.describe())
        
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV or Excel file to start.")

if __name__ == "__main__":
    main()
