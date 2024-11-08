import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Rest of the code remains the same, just remove the seaborn import and any seaborn-related code

class LPBFAnalyzer:
    def __init__(self, data):
        """Initialize the analyzer with the data."""
        if isinstance(data, str):
            self.df = pd.read_csv(data, header=1)
        else:
            data.seek(0)
            self.df = pd.read_csv(data, header=1)
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare the data for analysis."""
        # Remove unnamed columns
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        # Convert numeric columns
        for col in self.df.columns:
            try:
                # Replace any '#DIV/0!' with NaN
                self.df[col] = self.df[col].replace('#DIV/0!', np.nan)
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
            except:
                continue
        
        # Define mechanical properties to look for
        mech_props = ['UTS', 'YS', 'Elongation', 'hardness']
        self.available_props = [prop for prop in mech_props if prop in self.df.columns]
        
        if not self.available_props:
            st.error(f"No mechanical properties columns found. Looking for: {mech_props}")
            st.error("Available columns: " + ", ".join(self.df.columns.tolist()))
            raise ValueError("Required columns not found in dataset")

def main():
    st.set_page_config(page_title="LPBF Parameter Analyzer", layout="wide")
    
    st.title("LPBF AlSi10Mg Parameter Analysis Tool")
    
    uploaded_file = st.file_uploader("Upload your LPBF data CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            analyzer = LPBFAnalyzer(uploaded_file)
            
            # Add visualization options
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Distribution Analysis", "Parameter Relationships", "Process Window", "Build Direction Effects"]
            )
            
            if viz_type == "Distribution Analysis":
                show_distribution_analysis(analyzer)
            elif viz_type == "Parameter Relationships":
                show_parameter_relationships(analyzer)
            elif viz_type == "Process Window":
                show_process_window(analyzer)
            elif viz_type == "Build Direction Effects":
                show_direction_effects(analyzer)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Debug information:")
            try:
                df_debug = pd.read_csv(uploaded_file, nrows=5)
                st.write("First 5 rows:")
                st.write(df_debug)
            except Exception as debug_e:
                st.write(f"Error reading file: {str(debug_e)}")

def show_distribution_analysis(analyzer):
    st.header("Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        property_to_plot = st.selectbox(
            "Select Property to Analyze",
            analyzer.available_props
        )
        
        # Create combined histogram and box plot
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
            subplot_titles=(f"Distribution of {property_to_plot}", "Box Plot")
        )
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=analyzer.df[property_to_plot],
                name="Distribution",
                nbinsx=30,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add box plot
        fig.add_trace(
            go.Box(
                x=analyzer.df[property_to_plot],
                name="Box Plot",
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig)
    
    with col2:
        # Add statistical summary
        st.write("Statistical Summary:")
        stats_df = pd.DataFrame({
            'Statistic': [
                'Mean',
                'Median',
                'Std Dev',
                'Q1 (25%)',
                'Q3 (75%)',
                'IQR',
                'Min',
                'Max'
            ],
            'Value': [
                analyzer.df[property_to_plot].mean(),
                analyzer.df[property_to_plot].median(),
                analyzer.df[property_to_plot].std(),
                analyzer.df[property_to_plot].quantile(0.25),
                analyzer.df[property_to_plot].quantile(0.75),
                analyzer.df[property_to_plot].quantile(0.75) - analyzer.df[property_to_plot].quantile(0.25),
                analyzer.df[property_to_plot].min(),
                analyzer.df[property_to_plot].max()
            ]
        })
        st.dataframe(stats_df)

def show_parameter_relationships(analyzer):
    st.header("Parameter Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_param = st.selectbox(
            "Select X Parameter",
            analyzer.df.select_dtypes(include=[np.number]).columns
        )
        
        y_param = st.selectbox(
            "Select Y Parameter",
            [col for col in analyzer.df.select_dtypes(include=[np.number]).columns if col != x_param]
        )
        
        color_by = st.selectbox(
            "Color by",
            ["None", "Direction"] + [col for col in analyzer.df.columns if col not in [x_param, y_param]]
        )
    
    # Create scatter plot with trend line
    if color_by == "None":
        fig = px.scatter(
            analyzer.df,
            x=x_param,
            y=y_param,
            trendline="ols",
            trendline_color_override="red"
        )
    else:
        fig = px.scatter(
            analyzer.df,
            x=x_param,
            y=y_param,
            color=color_by,
            trendline="ols"
        )
    
    fig.update_layout(
        title=f"{y_param} vs {x_param}",
        height=600,
        width=800
    )
    
    st.plotly_chart(fig)
    
    # Calculate and display correlation
    correlation = analyzer.df[x_param].corr(analyzer.df[y_param])
    st.write(f"Correlation coefficient: {correlation:.3f}")

def show_process_window(analyzer):
    st.header("Process Window Analysis")
    
    # Get available process parameters
    process_params = [col for col in analyzer.df.columns if col in 
                     ["power", "speed", "Hatch", "thickness", "p/v"]]
    
    if len(process_params) < 2:
        st.warning("Not enough process parameters found in the dataset")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_process = st.selectbox(
            "Select X Process Parameter",
            process_params
        )
        
        y_process = st.selectbox(
            "Select Y Process Parameter",
            [p for p in process_params if p != x_process]
        )
        
        color_property = st.selectbox(
            "Color by Property",
            analyzer.available_props
        )

    # Convert color property to numeric, removing any units and handling NaN
    try:
        color_values = pd.to_numeric(
            analyzer.df[color_property].str.replace('Mpa', '').str.replace('HV', ''),
            errors='coerce'
        )
    except:
        # If direct conversion failed, try converting the column as is
        color_values = pd.to_numeric(analyzer.df[color_property], errors='coerce')
    
    # Create process window plot
    fig = go.Figure()
    
    # Filter out rows where any of the required values are NaN
    mask = ~(analyzer.df[x_process].isna() | 
             analyzer.df[y_process].isna() | 
             color_values.isna())
    
    # Add scatter plot with only valid data points
    scatter = go.Scatter(
        x=analyzer.df[x_process][mask],
        y=analyzer.df[y_process][mask],
        mode='markers',
        marker=dict(
            size=10,
            color=color_values[mask],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=f"{color_property} (numeric value)"
            )
        ),
        text=[f"{color_property}: {val}" for val in analyzer.df[color_property][mask]],
        hovertemplate=
        f"<b>{x_process}</b>: %{{x}}<br>" +
        f"<b>{y_process}</b>: %{{y}}<br>" +
        f"<b>{color_property}</b>: %{{text}}<extra></extra>"
    )
    
    fig.add_trace(scatter)
    
    # Update layout with more informative labels
    fig.update_layout(
        title=f"Process Window: Effect on {color_property}",
        xaxis_title=f"{x_process}",
        yaxis_title=f"{y_process}",
        height=600,
        width=800
    )
    
    # Add colorbar annotation
    fig.update_layout(
        annotations=[
            dict(
                x=1.2,
                y=0.5,
                xref="paper",
                yref="paper",
                text=f"Color shows {color_property} values",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    st.plotly_chart(fig)
    
    # Add statistical summary
    with st.expander("View Statistical Summary"):
        st.write("Summary Statistics:")
        summary_df = pd.DataFrame({
            'Parameter': [x_process, y_process, color_property],
            'Mean': [
                analyzer.df[x_process].mean(),
                analyzer.df[y_process].mean(),
                color_values.mean()
            ],
            'Std Dev': [
                analyzer.df[x_process].std(),
                analyzer.df[y_process].std(),
                color_values.std()
            ],
            'Min': [
                analyzer.df[x_process].min(),
                analyzer.df[y_process].min(),
                color_values.min()
            ],
            'Max': [
                analyzer.df[x_process].max(),
                analyzer.df[y_process].max(),
                color_values.max()
            ]
        })
        st.dataframe(summary_df)
        
        # Add correlation information
        st.write("Correlations:")
        corr_matrix = pd.DataFrame({
            x_process: [1, 
                       analyzer.df[x_process].corr(analyzer.df[y_process]),
                       analyzer.df[x_process].corr(color_values)],
            y_process: [analyzer.df[x_process].corr(analyzer.df[y_process]),
                       1,
                       analyzer.df[y_process].corr(color_values)],
            color_property: [analyzer.df[x_process].corr(color_values),
                           analyzer.df[y_process].corr(color_values),
                           1]
        }, index=[x_process, y_process, color_property])
        st.dataframe(corr_matrix.round(3))

def show_direction_effects(analyzer):
    st.header("Build Direction Effects")
    
    if 'Direction' not in analyzer.df.columns:
        st.warning("No build direction information in dataset")
        return
    
    property_name = st.selectbox(
        "Select Property",
        analyzer.available_props
    )
    
    # Create violin plot with individual points
    fig = go.Figure()
    
    for direction in analyzer.df['Direction'].unique():
        if pd.notna(direction):  # Only process non-NaN directions
            # Add violin plot
            fig.add_trace(go.Violin(
                x=analyzer.df[analyzer.df['Direction'] == direction]['Direction'],
                y=analyzer.df[analyzer.df['Direction'] == direction][property_name],
                name=str(direction),
                box_visible=True,
                meanline_visible=True,
                points='all'
            ))
    
    fig.update_layout(
        title=f"{property_name} Distribution by Build Direction",
        xaxis_title="Build Direction",
        yaxis_title=property_name,
        height=600,
        width=800,
        violinmode='group'
    )
    
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
