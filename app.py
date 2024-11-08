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
    
    # Convert columns with better error handling
    self.df['YS'] = self.df['YS'].apply(safe_numeric_convert)
    self.df['power'] = self.df['power'].apply(safe_numeric_convert)
    self.df['speed'] = self.df['speed'].apply(safe_numeric_convert)
    
    # Create analysis dataframe with basic parameters
    basic_params = ['power', 'speed', 'YS']
    self.analysis_df = self.df[basic_params].copy()
    
    # Fill missing values with medians for analysis
    for param in basic_params:
        median_value = self.analysis_df[param].median()
        self.analysis_df[param].fillna(median_value, inplace=True)
    
    # Add additional parameters if they exist
    additional_params = ['Hatch', 'thickness', 'p/v', 'Direction']
    for param in additional_params:
        if param in self.df.columns:
            self.analysis_df[param] = pd.to_numeric(self.df[param], errors='coerce')
            if param != 'Direction':  # Don't fill Direction with median
                self.analysis_df[param].fillna(self.analysis_df[param].median(), inplace=True)
    
    # Calculate p/v if not exists
    if 'p/v' not in self.analysis_df.columns:
        self.analysis_df['p/v'] = self.analysis_df['power'] / self.analysis_df['speed']
    
    # Add default values for missing parameters
    if 'Hatch' not in self.analysis_df.columns:
        self.analysis_df['Hatch'] = 0.13
    if 'thickness' not in self.analysis_df.columns:
        self.analysis_df['thickness'] = 0.03
    
    # Store original data counts
    self.data_counts = {
        'total': len(self.df),
        'valid_ys': len(self.df[self.df['YS'].notna()]),
        'valid_power': len(self.df[self.df['power'].notna()]),
        'valid_speed': len(self.df[self.df['speed'].notna()]),
        'used_in_analysis': len(self.analysis_df)
    }
    
    # Calculate process windows based on actual data
    def get_robust_range(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        median = series.median()
        iqr = q3 - q1
        return {
            'min': max(q1 - 1.5 * iqr, series.min()),
            'max': min(q3 + 1.5 * iqr, series.max()),
            'optimal': median
        }
    
    self.process_windows = {
        'power': get_robust_range(self.analysis_df['power']),
        'speed': get_robust_range(self.analysis_df['speed']),
        'p/v': get_robust_range(self.analysis_df['p/v'])
    }
    
    # Display data overview
    st.write("\nData Overview:")
    st.write(f"Total data points: {self.data_counts['total']}")
    st.write(f"Points used in analysis: {self.data_counts['used_in_analysis']}")
    
    # Store data statistics for reference
    self.data_stats = self.analysis_df[['power', 'speed', 'YS']].describe()
    
def create_process_window_plot(self, current_power, current_speed):
    """Create enhanced process window visualization"""
    fig = go.Figure()
    
    # Add dataset points with improved handling
    valid_mask = (self.df['power'].notna() & 
                 self.df['speed'].notna() & 
                 self.df['YS'].notna())
    
    # Add all data points
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
        hovertemplate=(
            'Power: %{x:.1f}W<br>' +
            'Speed: %{y:.1f}mm/s<br>' +
            'YS: %{marker.color:.1f}MPa<br>' +
            '<extra></extra>'
        )
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
        hovertemplate=(
            'Power: %{x:.1f}W<br>' +
            'Speed: %{y:.1f}mm/s<br>' +
            '<extra></extra>'
        )
    ))
    
    # Add process window rectangle with transparency
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
    
    # Add optimal point
    fig.add_trace(go.Scatter(
        x=[self.process_windows['power']['optimal']],
        y=[self.process_windows['speed']['optimal']],
        mode='markers',
        name='Optimal Point',
        marker=dict(
            size=12,
            color='green',
            symbol='diamond'
        )
    ))
    
    # Update layout with better formatting
    fig.update_layout(
        title={
            'text': 'Process Window Analysis',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Power (W)',
        yaxis_title='Scan Speed (mm/s)',
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=800,
        height=600
    )
    
    return fig

def analyze_parameters(self, power, speed):
    """Enhanced parameter analysis with more nuanced feedback"""
    p_v_ratio = power/speed
    
    # Initialize feedback categories
    issues = []
    process_window = []
    problems = []
    
    # Check power with more nuanced ranges
    power_range = self.process_windows['power']
    if power < power_range['min']:
        severity = (power_range['min'] - power) / power_range['min']
        if severity > 0.2:
            issues.append("Power is significantly below optimal range")
        else:
            issues.append("Power is slightly below optimal range")
    elif power > power_range['max']:
        severity = (power - power_range['max']) / power_range['max']
        if severity > 0.2:
            issues.append("Power is significantly above optimal range")
        else:
            issues.append("Power is slightly above optimal range")
    
    # Similar checks for speed and p/v ratio...
    # [Previous analyze_parameters code continues...]

    return {
        'issues': issues,
        'process_window': process_window,
        'problems': problems
    }
