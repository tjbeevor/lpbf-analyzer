import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats

class LPBFAnalyzer:
    # ... [Previous class methods remain the same until show_overview] ...

def show_overview(analyzer):
    st.header("Dataset Overview")
    
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

def show_distribution_analysis(analyzer):
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
    st.subheader("Parameter Relationships")
    
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
    st.subheader("Process Window Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_process = st.selectbox(
            "Select X Process Parameter",
            ["power", "speed", "Hatch", "thickness", "p/v"]
        )
        
        y_process = st.selectbox(
            "Select Y Process Parameter",
            [p for p in ["power", "speed", "Hatch", "thickness", "p/v"] if p != x_process]
        )
        
        color_property = st.selectbox(
            "Color by Property",
            analyzer.available_props
        )
    
    # Create process window plot
    fig = go.Figure()
    
    # Add scatter plot
    scatter = go.Scatter(
        x=analyzer.df[x_process],
        y=analyzer.df[y_process],
        mode='markers',
        marker=dict(
            size=10,
            color=analyzer.df[color_property],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_property)
        ),
        text=analyzer.df[color_property],
        hovertemplate=
        f'<b>{x_process}</b>: %{{x}}<br>' +
        f'<b>{y_process}</b>: %{{y}}<br>' +
        f'<b>{color_property}</b>: %{{text}}<extra></extra>'
    )
    
    fig.add_trace(scatter)
    
    fig.update_layout(
        title=f"Process Window: {color_property} Distribution",
        xaxis_title=x_process,
        yaxis_title=y_process,
        height=600,
        width=800
    )
    
    st.plotly_chart(fig)

def show_direction_effects(analyzer):
    st.subheader("Build Direction Effects")
    
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
        # Add violin plot
        fig.add_trace(go.Violin(
            x=analyzer.df[analyzer.df['Direction'] == direction]['Direction'],
            y=analyzer.df[analyzer.df['Direction'] == direction][property_name],
            name=direction,
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
    
    # Add statistical comparison
    if len(analyzer.df['Direction'].unique()) >= 2:
        st.write("Statistical Comparison between Directions:")
        
        directions = analyzer.df['Direction'].unique()
        stats_results = []
        
        for i in range(len(directions)):
            for j in range(i+1, len(directions)):
                dir1, dir2 = directions[i], directions[j]
                group1 = analyzer.df[analyzer.df['Direction'] == dir1][property_name].dropna()
                group2 = analyzer.df[analyzer.df['Direction'] == dir2][property_name].dropna()
                
                if len(group1) > 0 and len(group2) > 0:
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    stats_results.append({
                        'Comparison': f'{dir1} vs {dir2}',
                        't-statistic': t_stat,
                        'p-value': p_val
                    })
        
        if stats_results:
            st.dataframe(pd.DataFrame(stats_results))
