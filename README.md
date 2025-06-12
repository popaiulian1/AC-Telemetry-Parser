# AC Telemetry Parser
AC Telemetry Parser is a Python tool for analyzing and visualizing telemetry data from racing simulations (primarily Assetto Corsa). It transforms raw CSV telemetry exports from MoTeC into comprehensive visualizations that help drivers analyze their performance.  

## Features
- **Time Series Plots** - Visualize speed, RPM, gear changes, and pedal inputs over time  
- **Track Map Visualization** - See your racing line with speed-based coloring and direction indicators  
- **Driver Input Analysis** - Analyze throttle, brake, and steering behaviors with speed correlation  
- **Vehicle Dynamics Analysis** - View G-G diagrams, lateral/longitudinal forces, and chassis movements  
- **Interactive Dashboard** - Explore all telemetry data through an interactive HTML dashboard  

## Screenshots
### Track Map :
![Track Map](https://github.com/popaiulian1/AC-Telemetry-Parser/tree/main/plots/20250612_160823_track_map.png)  

### Time series analysis :
![Time series analysis](https://github.com/popaiulian1/AC-Telemetry-Parser/tree/main/plots/20250612_160823_time_series_plots.png)  

### Driver inputs :
![Driver inputs](https://github.com/popaiulian1/AC-Telemetry-Parser/tree/main/plots/20250612_160823_driver_inputs.png)  

### Vehicle dynamics :
![Vehicle dynamics](https://github.com/popaiulian1/AC-Telemetry-Parser/tree/main/plots/20250612_160823_vehicle_dynamics.png)  

## Requirements
- Python 3.7+  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- plotly  

## Instalation
1. Clone this repository :  

```bash
git clone https://github.com/popaiulian1/AC-Telemetry-Parser.git
cd AC-Telemetry-Parser
```

1. Install required dependencies :  

```bash
pip install pandas numpy matplotlib seaborn plotly
```

## Usage
1. Place your MoTeC CSV telemetry export files in the project directory.  
1. Run the script with your telemetry file:  
```bash
python data_plotting.py
```
1. By default the script processes `test_trial_lap.csv` and saves visualizations to the `plots` folder.  
1. To process a different file, modify the last lines in `data_plotting.py` :
```
df, metadata = load_motec_csv('your_telemetry_file.csv') <- This line here
plot_time_series(df, save_dir='plots')
plot_track_map(df, save_dir='plots')
plot_driver_inputs(df, save_dir='plots')
plot_vehicle_dynamics(df, save_dir='plots')
dashboard_fig = create_telemetry_dashboard(df, metadata, save_dir='plots')
```

## Interactive Dashboard
The script also generates an interactive HTML dashboard that combines all visualizations into a single interface. Open the generated HTML file in your browser to explore the telemetry data interactively.  
  
The interactive dashboard includes:
- Track map with speed data
- G-G diagram
- Speed and RPM plots with gear indicators
- Vehicle dynamics (roll/pitch angles)
- Driver inputs throughout the lap

## Data Format
The parser expects MoTeC CSV exports with:

- Metadata rows at the top (Vehicle, Venue, Driver info)
- Column headers including Time, Distance, Speed, etc.
- Numeric telemetry data in subsequent rows

## Customization
The visualization themes and colors can be customized by modifying the `set_custom_dark_theme()` function in `data_plotting.py`.