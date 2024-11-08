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
        """Clean and prepare the data with more inclusive handling"""
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Function to safely convert values
        def safe_numeric_convert(value):
            if pd.isna(value):
                return np.nan
            try:
                # Handle string values
                if isinstance(value, str):
                    # Remove common units and special characters
                    value = value.replace('Mpa', '').replace('HV', '').strip()
                    # Handle division by zero markers
                    if value in ['#DIV/0!', '#NUM!', 'inf', '-inf']:
                        return np.nan
                return float(value)
            except:
                return np.nan
        
        # Show initial data info
        st.write(f"Initial data points: {len(self.df)}")
        
        # Convert columns
        for col in ['YS', 'power', 'speed']:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(safe_numeric_convert)
        
        # Create analysis dataframe
        self.analysis_df = self.df.copy()
        
        # Calculate p/v ratio
        self.analysis_df['p/v'] = self.analysis_df['power'] / self.analysis_df['speed']
        
        # Set process windows based on valid data
        valid_mask = (self.analysis_df['power'].notna() & 
                     self.analysis_df['speed'].notna())
        
        self.process_windows = {
            'power': {
                'min': self.analysis_df[valid_mask]['power'].quantile(0.25),
                'max': self.analysis_df[valid_mask]['power'].quantile(0.75),
                'optimal': self.analysis_df[valid_mask]['power'].median()
            },
            'speed': {
                'min': self.analysis_df[valid_mask]['speed'].quantile(0.25),
                'max': self.analysis_df[valid_mask]['speed'].quantile(0.75),
                'optimal': self.analysis_df[valid_mask]['speed'].median()
            },
            'p/v': {
                'min': (self.analysis_df[valid_mask]['power'] / 
                       self.analysis_df[valid_mask]['speed']).quantile(0.25),
                'max': (self.analysis_df[valid_mask]['power'] / 
                       self.analysis_df[valid_mask]['speed']).quantile(0.75),
                'optimal': (self.analysis_df[valid_mask]['power'] / 
                          self.analysis_df[valid_mask]['speed']).median()
            }
        }
        
        st.write(f"Data points available for analysis: {sum(valid_mask)}")
    
    def train_model(self):
        """Train the prediction model"""
        # Prepare features
        features = ['power', 'speed', 'p/v']
        valid_mask = self.analysis_df[features + ['YS']].notna().all(axis=1)
        
        X = self.analysis_df[valid_mask][features]
        y = self.analysis_df[valid_mask]['YS']
        
        if len(X) < 10:
            st.error("Insufficient data for model training")
            return
        
        # Split and train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate metrics
        self.model_metrics = {
            'train_score': r2_score(self.y_train, self.model.predict(self.X_train)),
            'test_score': r2_score(self.y_test, self.model.predict(self.X_test)),
            'mae': mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        }
    
    def predict_strength(self, power, speed):
        """Predict yield strength for given parameters"""
        p_v = power/speed
        features = pd.DataFrame([[power, speed, p_v]], 
                              columns=['power', 'speed', 'p/v'])
        return self.model.predict(features)[0]
    
    def analyze_parameters(self, power, speed):
        """Analyze the given parameters"""
        p_v_ratio = power/speed
        
        issues = []
        process_window = []
        problems = []
        
        # Analysis logic
        if power < self.process_windows['power']['min']:
            issues.append("Power is below optimal range")
            problems.append("Insufficient energy for complete melting")
        elif power > self.process_windows['power']['max']:
            issues.append("Power is above optimal range")
            problems.append("Risk of keyholing defects")
        
        if speed < self.process_windows['speed']['min']:
            issues.append("Speed is below optimal range")
            problems.append("Excessive heat accumulation")
        elif speed > self.process_windows['speed']['max']:
            issues.append("Speed is above optimal range")
            problems.append("Insufficient melting likely")
        
        # Process window information
        process_window.extend([
            f"Optimal power range: {self.process_windows['power']['min']:.0f}-{self.process_windows['power']['max']:.0f}W",
            f"Optimal speed range: {self.process_windows['speed']['min']:.0f}-{self.process_windows['speed']['max']:.0f}mm/s",
            f"Current P/V ratio: {p_v_ratio:.2f} J/mm"
        ])
        
        return {
            'issues': issues,
            'process_window': process_window,
            'problems': problems
        }
    
    def create_process_window_plot(self, current_power, current_speed):
        """Create process window visualization"""
        fig = go.Figure()
        
        # Plot all valid data points
        valid_mask = (self.df['power'].notna() & 
                     self.df['speed'].notna() & 
                     self.df['YS'].notna())
        
        fig.add_trace(go.Scatter(
            x=self.df[valid_mask]['power'],
            y=self.df[valid_mask]['speed'],
            mode='markers',
            name='Dataset Points',
            marker=dict(
                size=8,
                color=self.df[valid_mask]['YS'],
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
            )
        ))
        
        # Add process window
        fig.add_shape(
            type="rect",
            x0=self.process_windows['power']['min'],
            y0=self.process_windows['speed']['min'],
            x1=self.process_windows['power']['max'],
            y1=self.process_windows['speed']['max'],
            line=dict(color="rgba(0,255,0,0.5)"),
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
            
            predicted_ys = predictor.predict_strength(power, speed)
            analysis = predictor.analyze_parameters(power, speed)
            
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
            
            st.plotly_chart(predictor.create_process_window_plot(power, speed))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Debug Information:")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, header=1)
                st.write("Columns in uploaded file:", df.columns.tolist())

if __name__ == "__main__":
    main()
