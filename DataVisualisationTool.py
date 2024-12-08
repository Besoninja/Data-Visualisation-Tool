import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from visualisation_catalog import get_suitable_visualisations, get_column_types

def main():
    st.title("Advanced Data Visualisation Tool")
    
    uploaded_file = st.sidebar.file_uploader("Upload Data", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # Data Overview
            with st.expander("Data Overview"):
                st.dataframe(df.head())
                st.write("Data Types:", df.dtypes)
            
            # Column Selection
            st.sidebar.header("Select Columns")
            selected_columns = st.sidebar.multiselect(
                "Choose columns to visualise",
                options=df.columns.tolist(),
                default=df.select_dtypes(include=['number']).columns[:2].tolist()
            )
            
            if selected_columns:
                # Get suitable visualisations
                vis_options = get_suitable_visualisations(df, selected_columns)
                
                if vis_options:
                    # Create tabs for different visualisation categories
                    tabs = st.tabs(list(vis_options.keys()))
                    
                    for tab, (category, vis_types) in zip(tabs, vis_options.items()):
                        with tab:
                            st.header(category)
                            
                            # Let user select visualisation type within category
                            vis_names = [vis.name for vis in vis_types]
                            selected_vis = st.selectbox(
                                f"Select {category} visualisation",
                                options=vis_names,
                                key=f"vis_select_{category}"
                            )
                            
                            # Get the selected visualisation type
                            vis_type = next(vis for vis in vis_types if vis.name == selected_vis)
                            
                            # Display visualisation description
                            st.info(vis_type.description)
                            
                            # Create visualisation
                            if vis_type.plotly_function:
                                try:
                                    fig = vis_type.plotly_function(
                                        df,
                                        x=selected_columns[0],
                                        y=selected_columns[1] if len(selected_columns) > 1 else None,
                                        color=selected_columns[2] if len(selected_columns) > 2 else None
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating visualisation: {str(e)}")
                else:
                    st.warning("No suitable visualisations found for the selected columns. Try selecting different columns.")
            else:
                st.info("Please select columns to visualise.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV or Excel file to begin.")

if __name__ == "__main__":
    main()
