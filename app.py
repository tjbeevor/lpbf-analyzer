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
    # ... [Previous analyzer code remains the same] ...

def main():
    st.set_page_config(page_title="LPBF Parameter Analyzer", layout="wide")
    
    st.title("LPBF AlSi10Mg Parameter Analysis Tool")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your LPBF data CSV", type="csv")
    
    if uploaded_file is not None:
        analyzer = LPBFAnalyzer(uploaded_file)
        
        # Sidebar for navigation
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Overview", "Heat Treatment Analysis", "Process Parameter Optimization", 
             "Build Direction Analysis", "Property Prediction"]
        )
        
        if analysis_type == "Overview":
            show_overview(analyzer)
        
        elif analysis_type == "Heat Treatment Analysis":
            show_heat_treatment_analysis(analyzer)
            
        elif analysis_type == "Process Parameter Optimization":
            show_process_optimization(analyzer)
            
        elif analysis_type == "Build Direction Analysis":
            show_build_direction_analysis(analyzer)
            
        elif analysis_type == "Property Prediction":
            show_property_prediction(analyzer)

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

def show_build_direction_analysis(analyzer):
    st.header("Build Direction Analysis")
    
    results = analyzer.analyze_build_direction_effects()
    
    properties = ["UTS", "YS", "Elongation"]
    
    for prop in properties:
        fig = px.box(
            analyzer.df,
            x="Direction",
            y=prop,
            title=f"{prop} by Build Direction"
        )
        st.plotly_chart(fig)
        
        if prop in results:
            st.write(f"Statistical Analysis for {prop}:")
            st.write(f"T-statistic: {results[prop]['t_statistic']:.3f}")
            st.write(f"P-value: {results[prop]['p_value']:.3f}")

def show_property_prediction(analyzer):
    st.header("Property Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        power = st.slider("Laser Power (W)", 100, 500, 350)
        speed = st.slider("Scan Speed (mm/s)", 500, 2000, 1000)
        hatch = st.slider("Hatch Spacing (mm)", 0.05, 0.3, 0.17)
        thickness = st.slider("Layer Thickness (mm)", 0.02, 0.06, 0.03)
        pv = st.slider("P/V Ratio", 0.1, 0.5, 0.3)
    
    if st.button("Predict Properties"):
        predictions = analyzer.predict_properties(
            [power, speed, hatch, thickness, pv]
        )
        
        with col2:
            st.subheader("Predicted Properties")
            for prop, value in predictions.items():
                st.write(f"{prop}: {value:.2f}")
            
            # Calculate quality score
            quality = analyzer.quality_score(
                predictions['UTS'],
                predictions['YS'],
                predictions['Elongation']
            )
            st.write(f"Quality Score: {quality:.2f}")

if __name__ == "__main__":
    main()
