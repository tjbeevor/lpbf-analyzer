# lpbf-analyzer
# LPBF Parameter Analyzer

A Streamlit web application for analyzing Laser Powder Bed Fusion (LPBF) process parameters and their effects on mechanical properties.

## Features
- Interactive parameter analysis
- Heat treatment effect visualization
- Process parameter optimization
- Build direction analysis
- Property prediction

## Usage
1. Upload your LPBF data CSV file
2. Select analysis type from the sidebar
3. Adjust parameters as needed
4. View results and download reports

## Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/lpbf-analyzer.git

# Install requirements
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

## Data Format
The application expects a CSV file with the following columns:
- Heat Treatment Parameters (solution temp, time, etc.)
- Material Properties (UTS, YS, Elongation, etc.)
- Process Parameters (power, speed, hatch spacing, etc.)
- Build Direction

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
[MIT](https://choosealicense.com/licenses/mit/)
