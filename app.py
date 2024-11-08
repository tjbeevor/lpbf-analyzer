import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LPBFAnalyzer:
    def __init__(self, data):
        """Initialize the analyzer with the data."""
        # Read CSV with first row as header, skipping the category row
        if isinstance(data, str):
            self.df = pd.read_csv(data, header=1)
        else:
            # Reset the file pointer to the beginning
            data.seek(0)
            self.df = pd.read_csv(data, header=1)
        
        self.clean_data()
    
    def clean_data(self):
        """Clean and prepare the data for analysis."""
        # Show available columns for debugging
        st.write("Available columns:", self.df.columns.tolist())
        
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

    def analyze_heat_treatment_effects(self, property_name):
        """Analyze the effect of heat treatment on a specific property."""
        if 'solution temp' not in self.df.columns:
            return "Solution temperature column not found in dataset"
        if property_name not in self.df.columns:
            return f"{property_name} not found in dataset"
        
        # Remove rows where either solution temp or the property is NaN
        valid_data = self.df[['solution temp', property_name]].dropna()
        
        if len(valid_data) > 0:
            corr = valid_data['solution temp'].corr(valid_data[property_name])
            return f"Correlation between solution temperature and {property_name}: {corr:.3f}"
        else:
            return "Not enough valid data points for correlation analysis"
    
    def process_parameter_optimization(self, target_property):
        """Optimize process parameters for a target property using machine learning."""
        # Look for process parameters in the dataset
        features = ['power', 'speed', 'Hatch', 'thickness', 'p/v']
        available_features = [f for f in features if f in self.df.columns]
        
        if not available_features:
            st.warning(f"No process parameters found. Looking for: {features}")
            return pd.DataFrame({'feature': ['No data'], 'importance': [0]})
        
        # Prepare data
        X = self.df[available_features].copy()
        y = self.df[target_property].copy()
        
        # Remove rows with NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 10:
            st.warning("Not enough valid data points for analysis")
            return pd.DataFrame({'feature': available_features, 'importance': [0]*len(available_features)})
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance

def main():
    st.set_page_config(page_title="LPBF Parameter Analyzer", layout="wide")
    
    st.title("LPBF AlSi10Mg Parameter Analysis Tool")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your LPBF data CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Show raw data first for debugging
            st.write("Preview of raw data:")
            df_raw = pd.read_csv(uploaded_file, header=1, nrows=5)
            st.write(df_raw)
            
            analyzer = LPBFAnalyzer(uploaded_file)
            
            # Sidebar for navigation
            analysis_type = st.sidebar.selectbox(
                "Select Analysis Type",
                ["Overview", "Heat Treatment Analysis", "Process Parameter Optimization"]
            )
            
            if analysis_type == "Overview":
                show_overview(analyzer)
            
            elif analysis_type == "Heat Treatment Analysis":
                show_heat_treatment_analysis(analyzer)
                
            elif analysis_type == "Process Parameter Optimization":
                show_process_optimization(analyzer)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Debug information:")
            st.write("File contents preview:")
            try:
                df_debug = pd.read_csv(uploaded_file, nrows=5)
                st.write("First 5 rows of raw data:")
                st.write(df_debug)
                st.write("Columns found:", df_debug.columns.tolist())
            except Exception as debug_e:
                st.write(f"Error reading file: {str(debug_e)}")

def show_overview(analyzer):
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Summary")
        st.write(analyzer.df.describe())
        
    with col2:
        st.subheader("Data Distribution")
        if analyzer.available_props:
            property_to_plot = st.selectbox(
                "Select Property to Visualize",
                analyzer.available_props
            )
            
            fig = px.histogram(analyzer.df, x=property_to_plot)
            st.plotly_chart(fig)
        else:
            st.write("No mechanical properties available for visualization")

def show_heat_treatment_analysis(analyzer):
    st.header("Heat Treatment Analysis")
    
    if not analyzer.available_props:
        st.warning("No mechanical properties available for analysis")
        return
        
    property_name = st.selectbox(
        "Select Property to Analyze",
        analyzer.available_props
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'solution temp' in analyzer.df.columns:
            # Remove rows where either solution temp or property is NaN
            valid_data = analyzer.df[['solution temp', property_name, 'Direction']].dropna()
            
            if len(valid_data) > 0:
                fig = px.scatter(
                    valid_data,
                    x="solution temp",
                    y=property_name,
                    color="Direction",
                    title=f"Effect of Solution Treatment Temperature on {property_name}"
                )
                st.plotly_chart(fig)
            else:
                st.warning("No valid data points for plotting")
        else:
            st.warning("Solution temperature column not found in dataset")
        
    with col2:
        if st.button("Calculate Correlations"):
            corr = analyzer.analyze_heat_treatment_effects(property_name)
            st.write(corr)

def show_process_optimization(analyzer):
    st.header("Process Parameter Optimization")
    
    if not analyzer.available_props:
        st.warning("No mechanical properties available for analysis")
        return
        
    target_property = st.selectbox(
        "Select Target Property",
        analyzer.available_props
    )
    
    importance = analyzer.process_parameter_optimization(target_property)
    
    if len(importance) > 0 and 'No data' not in importance['feature'].values:
        fig = px.bar(
            importance,
            x='feature',
            y='importance',
            title=f"Feature Importance for {target_property}"
        )
        st.plotly_chart(fig)
        
        st.write("Optimization Results:", importance)
    else:
        st.warning("No valid process parameters found for analysis")

if __name__ == "__main__":
    main()
