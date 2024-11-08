import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

class LPBFPredictor:
    def __init__(self, data_file):
        """Initialize the predictor with data file"""
        self.df = pd.read_csv(data_file, header=1)
        self.clean_data()
        self.train_model()
        
    def clean_data(self):
        """Clean and prepare the data"""
        # Clean column names (remove trailing spaces)
        self.df.columns = self.df.columns.str.strip()
        
        # Convert columns to numeric
        self.df['YS'] = pd.to_numeric(self.df['YS'].astype(str).str.replace('Mpa', ''), errors='coerce')
        self.df['power'] = pd.to_numeric(self.df['power'], errors='coerce')
        self.df['speed'] = pd.to_numeric(self.df['speed'], errors='coerce')
        
        # Debug print
        st.write("Available columns:", self.df.columns.tolist())
        
        # Create analysis dataframe with basic parameters
        basic_params = ['power', 'speed', 'YS']
        self.analysis_df = self.df[basic_params].copy()
        
        # Add additional parameters if they exist
        additional_params = ['Hatch', 'thickness', 'p/v', 'Direction']
        for param in additional_params:
            if param in self.df.columns:
                self.analysis_df[param] = pd.to_numeric(self.df[param], errors='coerce')
        
        # If p/v doesn't exist, calculate it from power and speed
        if 'p/v' not in self.analysis_df.columns:
            self.analysis_df['p/v'] = self.analysis_df['power'] / self.analysis_df['speed']
        
        # Add default values for missing parameters
        if 'Hatch' not in self.analysis_df.columns:
            self.analysis_df['Hatch'] = 0.13  # typical value
        if 'thickness' not in self.analysis_df.columns:
            self.analysis_df['thickness'] = 0.03  # typical value
            
        self.analysis_df = self.analysis_df.dropna(subset=['power', 'speed', 'YS'])
        
        # Calculate typical process windows
        self.process_windows = {
            'power': {'min': 300, 'max': 400, 'optimal': 350},
            'speed': {'min': 800, 'max': 1300, 'optimal': 1000},
            'p/v': {'min': 0.3, 'max': 0.5, 'optimal': 0.4}
        }
    
    def train_model(self):
        """Train the prediction model"""
        # Use only numeric columns for prediction
        numeric_columns = self.analysis_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col != 'YS']
        
        X = self.analysis_df[feature_columns]
        y = self.analysis_df['YS']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate model performance metrics
        self.model_metrics = {
            'train_score': r2_score(self.y_train, self.model.predict(self.X_train)),
            'test_score': r2_score(self.y_test, self.model.predict(self.X_test)),
            'mae': mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        }
        
        # Store feature names for prediction
        self.feature_names = feature_columns
    
    def predict_strength(self, power, speed):
        """Predict yield strength for given parameters"""
        # Create prediction input
        pred_input = pd.DataFrame(columns=self.feature_names)
        pred_input.loc[0, 'power'] = power
        pred_input.loc[0, 'speed'] = speed
        pred_input.loc[0, 'p/v'] = power/speed
        
        # Fill in default values for other features if they exist
        if 'Hatch' in self.feature_names:
            pred_input.loc[0, 'Hatch'] = self.analysis_df['Hatch'].median()
        if 'thickness' in self.feature_names:
            pred_input.loc[0, 'thickness'] = self.analysis_df['thickness'].median()
        
        # Make prediction
        prediction = self.model.predict(pred_input)[0]
        return prediction
    
    def analyze_parameters(self, power, speed):
        """Analyze the given parameters and provide feedback"""
        p_v_ratio = power/speed
        
        # Initialize feedback categories
        issues = []
        process_window = []
        problems = []
        
        # Check power
        if power < self.process_windows['power']['min']:
            issues.append("Power is too low for optimal processing")
            problems.append("Insufficient energy for complete melting")
            problems.append("Poor layer-to-layer bonding likely")
        elif power > self.process_windows['power']['max']:
            issues.append("Power is higher than typical range")
            problems.append("Risk of keyholing defects")
            problems.append("Potential for excessive heat input")
        
        # Check speed
        if speed < self.process_windows['speed']['min']:
            issues.append("Scan speed is too low")
            problems.append("Excessive heat accumulation possible")
            problems.append("Poor surface finish likely")
        elif speed > self.process_windows['speed']['max']:
            issues.append("Scan speed is too high")
            problems.append("Insufficient melting may occur")
            problems.append("Risk of lack-of-fusion defects")
        
        # Check P/V ratio
        if p_v_ratio < self.process_windows['p/v']['min']:
            process_window.append("P/V ratio is below optimal range")
            problems.append("Incomplete melting likely")
        elif p_v_ratio > self.process_windows['p/v']['max']:
            process_window.append("P/V ratio is above optimal range")
            problems.append("Excessive energy input")
        
        # Add process window information
        process_window.append(f"Optimal power range: {self.process_windows['power']['min']}-{self.process_windows['power']['max']}W")
        process_window.append(f"Optimal speed range: {self.process_windows['speed']['min']}-{self.process_windows['speed']['max']}mm/s")
        process_window.append(f"Optimal P/V ratio: {self.process_windows['p/v']['min']}-{self.process_windows['p/v']['max']}J/mm")
        
        return {
            'issues': issues,
            'process_window': process_window,
            'problems': problems
        }
    
    def create_process_window_plot(self, current_power, current_speed):
        """Create process window visualization"""
        fig = go.Figure()
        
        # Add dataset points
        fig.add_trace(go.Scatter(
            x=self.analysis_df['power'],
            y=self.analysis_df['speed'],
            mode='markers',
            name='Dataset Points',
            marker=dict(
                size=8,
                color=self.analysis_df['YS'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Yield Strength (MPa)')
            ),
            hovertemplate='Power: %{x}W<br>Speed: %{y}mm/s<br>YS: %{marker.color:.0f}MPa<extra></extra>'
        ))
        
        # Add current point
        fig.add_trace(go.Scatter(
            x=[current_power],
            y=[current_speed],
            mode='markers',
            name='Selected Parameters',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            hovertemplate='Power: %{x}W<br>Speed: %{y}mm/s<extra></extra>'
        ))
        
        # Add process window rectangle
        fig.add_shape(
            type="rect",
            x0=self.process_windows['power']['min'],
            y0=self.process_windows['speed']['min'],
            x1=self.process_windows['power']['max'],
            y1=self.process_windows['speed']['max'],
            line=dict(
                color="rgba(0,255,0,0.5)",
                width=2,
            ),
            fillcolor="rgba(0,255,0,0.1)"
        )
        
        fig.update_layout(
            title='Process Window Analysis',
            xaxis_title='Power (W)',
            yaxis_title='Scan Speed (mm/s)',
            hovermode='closest'
        )
        
        return fig

def main():
    st.set_page_config(page_title="LPBF Parameter Predictor", layout="wide")
    
    st.title("LPBF AlSi10Mg Parameter Predictor")
    
    uploaded_file = st.file_uploader("Upload LPBF data CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            predictor = LPBFPredictor(uploaded_file)
            
            st.sidebar.header("Parameter Input")
            power = st.sidebar.slider("Laser Power (W)", 100, 800, 350)
            speed = st.sidebar.slider("Scan Speed (mm/s)", 100, 2000, 1000)
            
            # Make prediction
            predicted_ys = predictor.predict_strength(power, speed)
            
            # Get analysis
            analysis = predictor.analyze_parameters(power, speed)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Prediction Results")
                st.metric("Predicted Yield Strength", f"{predicted_ys:.1f} MPa")
                
                st.subheader("Key Issues")
                if analysis['issues']:
                    for issue in analysis['issues']:
                        st.warning(issue)
                else:
                    st.success("Parameters are within typical ranges")
                
                st.subheader("Potential Problems")
                if analysis['problems']:
                    for problem in analysis['problems']:
                        st.error(problem)
                else:
                    st.success("No significant problems expected")
            
            with col2:
                st.subheader("Process Window Analysis")
                for info in analysis['process_window']:
                    st.info(info)
                
                st.subheader("Model Performance")
                st.write(f"Training R² Score: {predictor.model_metrics['train_score']:.3f}")
                st.write(f"Testing R² Score: {predictor.model_metrics['test_score']:.3f}")
                st.write(f"Mean Absolute Error: {predictor.model_metrics['mae']:.1f} MPa")
            
            # Show process window plot
            st.plotly_chart(predictor.create_process_window_plot(power, speed))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Debug Information:")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, header=1)
                st.write("Columns in uploaded file:", df.columns.tolist())

if __name__ == "__main__":
    main()
