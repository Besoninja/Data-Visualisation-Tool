# visualisation_catalog.py

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    GEOGRAPHIC = "geographic"
    TEXT = "text"

@dataclass
class VisualisationType:
    name: str
    description: str
    required_columns: Dict[DataType, int]  # e.g., {DataType.NUMERIC: 1, DataType.CATEGORICAL: 1}
    optional_columns: Dict[DataType, int]
    max_columns: int
    category: str
    plotly_function: Optional[callable] = None
    
def get_column_types(df: pd.DataFrame) -> Dict[str, DataType]:
    """Identify the data type of each column."""
    column_types = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = DataType.NUMERIC
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            column_types[column] = DataType.DATETIME
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            # Check if it might be geographic data
            if column.lower() in ['country', 'state', 'city', 'region', 'latitude', 'longitude', 'lat', 'lon']:
                column_types[column] = DataType.GEOGRAPHIC
            else:
                column_types[column] = DataType.CATEGORICAL
    return column_types

VISUALISATION_CATALOG = {
    "Basic": {
        "scatter": VisualisationType(
            name="Scatter Plot",
            description="Show relationship between two numeric variables",
            required_columns={DataType.NUMERIC: 2},
            optional_columns={DataType.CATEGORICAL: 1},
            max_columns=4,
            category="Graphs/Plots",
            plotly_function=px.scatter
        ),
        "line": VisualisationType(
            name="Line Graph",
            description="Show trends over a continuous variable",
            required_columns={DataType.NUMERIC: 1, DataType.NUMERIC: 1},
            optional_columns={DataType.CATEGORICAL: 1},
            max_columns=3,
            category="Graphs/Plots",
            plotly_function=px.line
        ),
        "bar": VisualisationType(
            name="Bar Chart",
            description="Compare quantities across categories",
            required_columns={DataType.NUMERIC: 1, DataType.CATEGORICAL: 1},
            optional_columns={DataType.CATEGORICAL: 1},
            max_columns=3,
            category="Graphs/Plots",
            plotly_function=px.bar
        ),
    },
    "Statistical": {
        "box": VisualisationType(
            name="Box Plot",
            description="Show distribution and outliers",
            required_columns={DataType.NUMERIC: 1},
            optional_columns={DataType.CATEGORICAL: 1},
            max_columns=2,
            category="Graphs/Plots",
            plotly_function=px.box
        ),
        "violin": VisualisationType(
            name="Violin Plot",
            description="Show probability density of data",
            required_columns={DataType.NUMERIC: 1},
            optional_columns={DataType.CATEGORICAL: 1},
            max_columns=2,
            category="Graphs/Plots",
            plotly_function=px.violin
        ),
        "histogram": VisualisationType(
            name="Histogram",
            description="Show distribution of a numeric variable",
            required_columns={DataType.NUMERIC: 1},
            optional_columns={},
            max_columns=1,
            category="Graphs/Plots",
            plotly_function=px.histogram
        ),
    },
    "Relationships": {
        "heatmap": VisualisationType(
            name="Heatmap",
            description="Show correlation between numeric variables",
            required_columns={DataType.NUMERIC: 2},
            optional_columns={},
            max_columns=10,
            category="Tables",
            plotly_function=px.imshow
        ),
        "parallel_coordinates": VisualisationType(
            name="Parallel Coordinates",
            description="Show patterns between multiple variables",
            required_columns={DataType.NUMERIC: 3},
            optional_columns={DataType.CATEGORICAL: 1},
            max_columns=10,
            category="Graphs/Plots",
            plotly_function=px.parallel_coordinates
        ),
    },
    "Time Series": {
        "time_series": VisualisationType(
            name="Time Series",
            description="Show changes over time",
            required_columns={DataType.DATETIME: 1, DataType.NUMERIC: 1},
            optional_columns={DataType.CATEGORICAL: 1},
            max_columns=3,
            category="Graphs/Plots",
            plotly_function=px.line
        ),
    },
    "Geographic": {
        "choropleth": VisualisationType(
            name="Choropleth Map",
            description="Show values on a map",
            required_columns={DataType.GEOGRAPHIC: 1, DataType.NUMERIC: 1},
            optional_columns={},
            max_columns=2,
            category="Maps/Geographical",
            plotly_function=px.choropleth
        ),
    }
}

def get_suitable_visualisations(df: pd.DataFrame, selected_columns: List[str]) -> Dict[str, List[VisualisationType]]:
    """
    Determine which visualisations are suitable for the selected columns.
    Returns a dictionary of visualisation categories and their suitable visualisation types.
    """
    column_types = get_column_types(df)
    selected_types = {col: column_types[col] for col in selected_columns}
    
    # Count the number of each type of column
    type_counts = {dtype: sum(1 for col_type in selected_types.values() if col_type == dtype) 
                  for dtype in DataType}
    
    suitable_vis = {}
    
    for category, vis_types in VISUALISATION_CATALOG.items():
        suitable_in_category = []
        for vis_name, vis_type in vis_types.items():
            # Check if we have enough columns of each required type
            requirements_met = True
            for req_type, req_count in vis_type.required_columns.items():
                if type_counts.get(req_type, 0) < req_count:
                    requirements_met = False
                    break
                    
            # Check if we're not exceeding max columns
            if len(selected_columns) > vis_type.max_columns:
                requirements_met = False
                
            if requirements_met:
                suitable_in_category.append(vis_type)
                
        if suitable_in_category:
            suitable_vis[category] = suitable_in_category
            
    return suitable_vis
