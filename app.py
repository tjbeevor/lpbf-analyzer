import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import base64
from io import StringIO

class LPBFAnalyzer:
    def __init__(self, data):
        """Initialize the analyzer with the data."""
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = pd.read_csv(data)
        self.clean_data()
    
    def clean_data(self):
        """Clean and prepare the data for analysis."""
        # Convert numeric columns from string to float
        numeric_cols = self.df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            except:
                continue
        
        # Remove rows where all mechanical properties are NaN
        mech_props = ['UTS', 'YS', 'Elongation', 'hardness']
        self.df = self.df.dropna(subset=mech_props, how='all')
    
    def analyze_heat_treatment_effects(self, property_name):
        """Analyze the effect of heat treatment on a specific property."""
        corr = self.df['solution temp'].corr(self.df[property_name])
        return f"Correlation between solution temperature and {property_name}: {corr:.3f}"
    
    def process_parameter_optimization(self, target_property):
        """Optimize process parameters for a target property using machine learning."""
        features = ['power', 'speed', 'Hatch', 'thickness', 'p/v']
        
        # Prepare data
        X = self.df[features].dropna()
        y = self.df[target_property].dropna()
        
        # Only use rows where we have both X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 10:  # Check if we have enough data
            return pd.DataFrame({'feature': features, 'importance': [0]*len(features)})
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': features,
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
            st.write("Please make sure your CSV file has the correct format.")

def show_overview(analyzer):
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Summary")
        st.write(analyzer.df.describe())
        
    with col2:
        st.subheader("Data Distribution")
        property_to_plot = st.selectbox(
            "Select Property to Visualize",
            ["UTS", "YS", "Elongation", "hardness"]
        )
        
        fig = px.histogram(analyzer.df, x=property_to_plot)
        st.plotly_chart(fig)

def show_heat_treatment_analysis(analyzer):
    st.header("Heat Treatment Analysis")
    
    property_name = st.selectbox(
        "Select Property to Analyze",
        ["UTS", "YS", "Elongation", "hardness"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            analyzer.df,
            x="solution temp",
            y=property_name,
            color="Direction",
            title=f"Effect of Solution Treatment Temperature on {property_name}"
        )
        st.plotly_chart(fig)
        
    with col2:
        if st.button("Calculate Correlations"):
            corr = analyzer.analyze_heat_treatment_effects(property_name)
            st.write(corr)

def show_process_optimization(analyzer):
    st.header("Process Parameter Optimization")
    
    target_property = st.selectbox(
        "Select Target Property",
        ["UTS", "YS", "Elongation"]
    )
    
    importance = analyzer.process_parameter_optimization(target_property)
    
    fig = px.bar(
        importance,
        x='feature',
        y='importance',
        title=f"Feature Importance for {target_property}"
    )
    st.plotly_chart(fig)
    
    st.write("Optimization Results:", importance)

if __name__ == "__main__":
    main()
