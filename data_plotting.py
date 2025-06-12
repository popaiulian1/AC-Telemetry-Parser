from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

def set_custom_dark_theme() -> dict[str, str]:
    """
    Set custom dark theme for all matplotlib plots.
    
    Configures a dark-themed visualization style with consistent colors
    for different telemetry components including speed, RPM, throttle, etc.
    
    Returns:
        dict[str, str]: Dictionary of predefined colors for different telemetry elements
    """
    plt.style.use('dark_background')
    
    dark_bg = '#0A0A0A'  
    grid_color = '#444444'  
    text_color = '#FFFFFF'  
    
    colors = {
        'speed': '#00FFFF',
        'rpm': '#FFA500',
        'gear': '#50FF78',
        'throttle': '#CCFF00',
        'brake': "#F31D1D",
        'steering': '#FFFFFF',
        'lateral_g': '#FF55FF',
        'long_g': '#FFFF00',
        'tire_temp': '#FF6347'
    }
    
    plt.rcParams.update({
        'figure.facecolor': dark_bg,
        'axes.facecolor': dark_bg,
        'axes.edgecolor': text_color,
        'axes.labelcolor': text_color,
        'axes.grid': True,
        'axes.grid.which': 'both',
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': grid_color,
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'text.color': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'figure.figsize': (12, 8),
        'savefig.facecolor': dark_bg,
        'savefig.edgecolor': dark_bg,
        'font.size': 12, 
        'axes.titlesize': 15,  
        'axes.labelsize': 13,  
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.facecolor': 'black',
        'legend.edgecolor': grid_color,
    })
    
    sns.set_style("darkgrid", {
        'axes.facecolor': dark_bg,
        'grid.color': grid_color,
        'axes.edgecolor': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'text.color': text_color,
        'grid.linestyle': '-',
    })
    
    return colors

colors = set_custom_dark_theme()
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.style.use('dark_background')
sns.set_style("darkgrid")

def load_motec_csv(filepath: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Parses telemetry data and metadata from a MoTeC CSV export file,
    handling the special format with metadata in header rows.
    
    Args:
        filepath (str): Path to the MoTeC CSV file
        
    Returns:
        tuple[pd.DataFrame, dict[str, str]]: Tuple containing:
            - DataFrame with all telemetry data
            - Dictionary with session metadata (track, vehicle, driver info)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_line = 0
    for i, line in enumerate(lines):
        if 'Time' in line and 'Distance' in line:
            header_line = i
            break

    metadata = {}
    for i in range(header_line):
        line = lines[i].strip().strip('"').split('","')
        if len(line) >= 2:
            if line[0] not in ['', ' ']:
                if '"' in line[1]:
                    value = line[1].split('"')[0].strip()
                else:
                    value = line[1].strip()
                metadata[line[0]] = value

    header = lines[header_line].strip().replace('"', '').split(',')
    
    data_lines = [line.strip() for line in lines[header_line+2:] if line.strip()]
    
    # Create a list of dictionaries for the data
    data = []
    for line in data_lines:
        values = line.strip('"').split('","')
        values = [val.replace(',', '.') for val in values]
        # Create a dictionary mapping column names to values
        row_dict = {header[i]: values[i] if i < len(values) else None 
                   for i in range(min(len(header), len(values)))}
        data.append(row_dict)
    
    df = pd.DataFrame(data)
    
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    print(f"Loaded data from {os.path.basename(filepath)}")
    print(f"Track: {metadata.get('Venue', 'Unknown')}")
    print(f"Vehicle: {metadata.get('Vehicle', 'Unknown')}")
    print(f"Driver: {metadata.get('Driver', 'Unknown')}")
    print(f"Data points: {len(df)}")
    print(f"Columns: {df.columns.tolist()[:5]} ... (total: {len(df.columns)})")
    
    return df, metadata

def plot_time_series(df: pd.DataFrame, save_dir: Optional[str] = None) -> plt.Figure:
    """
    Generates a 4-panel figure showing speed, RPM, gear, and throttle/brake
    positions over time, with consistent styling.
    
    Args:
        df (pd.DataFrame): Telemetry data with time-based information
        save_dir (Optional[str], optional): Directory to save the plot image. 
            If None, displays the plot instead. Defaults to None.
            
    Returns:
        plt.Figure: Matplotlib figure object containing the time series plots
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    df.plot(x='Time', y='Ground Speed', ax=axes[0], color=colors['speed'], linewidth=1.5)
    axes[0].set_ylabel('Speed (km/h)', color='white')
    axes[0].set_title('Speed vs Time', color='white')
    axes[0].grid(True)

    df.plot(x='Time', y='Engine RPM', ax=axes[1], color=colors['rpm'], linewidth=1.5)
    axes[1].set_ylabel('RPM', color='white')
    axes[1].set_title('Engine RPM vs Time', color='white')
    axes[1].grid(True)

    df.plot(x='Time', y='Gear', ax=axes[2], color=colors['gear'], linewidth=1.5)
    axes[2].set_ylabel('Gear', color='white')
    axes[2].set_title('Gear vs Time', color='white')
    axes[2].set_yticks(range(0, int(df['Gear'].max()) +2))
    axes[2].grid(True)

    df.plot(x='Time', y='Throttle Pos', ax=axes[3], color=colors['throttle'], linewidth=1.5)
    df.plot(x='Time', y='Brake Pos', ax=axes[3], color=colors['brake'], linewidth=2.0)
    axes[3].set_ylabel('Pedal Position (%)', color='white')
    axes[3].set_xlabel('Time (s)', color='white')
    axes[3].set_title('Throttle and Brake vs Time', color='white')
    axes[3].legend()
    axes[3].grid(True)

    for ax in axes:
        ax.set_facecolor('#121212')
        ax.grid(color='#444444', linestyle='--', linewidth=0.6)
        ax.tick_params(axis='both', colors='white')

        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_color('white')
        
        # Add a subtle box around the plot area
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
            spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, timestamp_str+'_time_series_plots.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig

def plot_track_map(df: pd.DataFrame, save_dir: Optional[str] = None) -> plt.Figure:
    """
    Creates a 2D track map from position data with speed-based coloring
    and direction indicators. The start/finish point is highlighted.
    
    Args:
        df (pd.DataFrame): Telemetry data with Car Coord X/Y and speed information
        save_dir (Optional[str], optional): Directory to save the plot image.
            If None, displays the plot instead. Defaults to None.
            
    Returns:
        plt.Figure: Matplotlib figure object containing the track map
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    

    scatter = ax.scatter(df['Car Coord X'], df['Car Coord Y'],
                         c=df['Ground Speed'], cmap='viridis', s=10, alpha=0.8)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Speed (km/h)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')

    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    N = len(df) // 10 # Reduce number of points for arrows
    for i in range (0, len(df), N):
        if i + 1 < len(df):
            ax.arrow(df['Car Coord X'].iloc[i], df['Car Coord Y'].iloc[i],
                     df['Car Coord X'].iloc[i + 1] - df['Car Coord X'].iloc[i],
                     df['Car Coord Y'].iloc[i + 1] - df['Car Coord Y'].iloc[i],
                     head_width=2, head_length=2, fc='white', ec='white')
            
    start_idx = df['Time'].idxmin()
    ax.plot(df.loc[start_idx, 'Car Coord X'], df.loc[start_idx, 'Car Coord Y'],
             'ro', markersize=12, label='Start/Finish')
    
    ax.set_title('Track Map Visualization with Speed', color='white')
    ax.set_xlabel('X position (m)', color='white')
    ax.set_ylabel('Y position (m)', color='white')
    ax.set_aspect('equal')
    ax.grid(True, color='white', linestyle='--', alpha=0.7)
    ax.legend()

    ax.set_facecolor('#121212')

    if ax.get_legend() is not None:
        for text in ax.get_legend().get_texts():
            text.set_color('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#FFFFFF')
        spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, timestamp_str+'_track_map.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()

    return fig

def plot_driver_inputs(df: pd.DataFrame, save_dir: Optional[str] = None) -> plt.Figure:
    """
    Analyze and visualize driver inputs (throttle, brake, steering).
    
    Creates a 2x2 grid of plots showing relationships between driver inputs
    and vehicle speed, as well as statistical distributions of inputs.
    
    Args:
        df (pd.DataFrame): Telemetry data with driver inputs and speed information
        save_dir (Optional[str], optional): Directory to save the plot image.
            If None, displays the plot instead. Defaults to None.
            
    Returns:
        plt.Figure: Matplotlib figure object containing the driver input analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    scatter1 = axes[0, 0].scatter(df['Ground Speed'], df['Throttle Pos'], 
                     c=df['Time'], cmap='plasma', alpha=0.8, s=15)
    axes[0, 0].set_title('Throttle vs Speed', color='white', fontsize=14)
    axes[0, 0].set_xlabel('Speed (km/h)', color='white', fontsize=12)
    axes[0, 0].set_ylabel('Throttle Position (%)', color='white', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Time (s)', color='white', fontsize=12)
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
    
    scatter2 = axes[0, 1].scatter(df['Ground Speed'], df['Brake Pos'], 
                     c=df['Time'], cmap='plasma', alpha=0.8, s=15)
    axes[0, 1].set_title('Brake vs Speed', color='white', fontsize=14)
    axes[0, 1].set_xlabel('Speed (km/h)', color='white', fontsize=12)
    axes[0, 1].set_ylabel('Brake Position (%)', color='white', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('Time (s)', color='white', fontsize=12)
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
    
    scatter3 = axes[1, 0].scatter(df['Ground Speed'], df['Steering Angle'], 
                     c=df['Time'], cmap='plasma', alpha=0.8, s=15)
    axes[1, 0].set_title('Steering vs Speed', color='white', fontsize=14)
    axes[1, 0].set_xlabel('Speed (km/h)', color='white', fontsize=12)
    axes[1, 0].set_ylabel('Steering Angle (deg)', color='white', fontsize=12)
    cbar3 = plt.colorbar(scatter3, ax=axes[1, 0])
    cbar3.set_label('Time (s)', color='white', fontsize=12)
    cbar3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')
    
    df_melt = pd.melt(df, id_vars=['Time'], value_vars=['Throttle Pos', 'Brake Pos'])
    sns.violinplot(x='variable', y='value', hue='variable', data=df_melt, ax=axes[1, 1], 
                  palette=['#CCFF00', '#FF3030'], linewidth=1, legend=False)
    axes[1, 1].set_title('Throttle and Brake Distribution', color='white', fontsize=14)
    axes[1, 1].set_xlabel('', color='white', fontsize=12)
    axes[1, 1].set_ylabel('Position (%)', color='white', fontsize=12)
    
    for ax in axes.flat:
        ax.set_facecolor('#0A0A0A')
        ax.grid(color='#333333', linestyle='-', linewidth=0.8, alpha=0.8) 
        

        for spine in ax.spines.values():
            spine.set_edgecolor('#FFFFFF') 
            spine.set_linewidth(1.0)
            
        ax.tick_params(colors='white', labelsize=11)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, timestamp_str+'_driver_inputs.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_vehicle_dynamics(df: pd.DataFrame, save_dir: Optional[str] = None) -> plt.Figure:
    """
    Analyze and visualize vehicle dynamics (lateral G, longitudinal G, etc.).
    
    Creates a 2x2 grid of plots showing g-forces, chassis movements,
    and their relationships with speed and driver inputs.
    
    Args:
        df (pd.DataFrame): Telemetry data with acceleration and chassis information
        save_dir (Optional[str], optional): Directory to save the plot image.
            If None, displays the plot instead. Defaults to None.
            
    Returns:
        plt.Figure: Matplotlib figure object containing the vehicle dynamics analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    scatter1 = axes[0, 0].scatter(df['Ground Speed'], df['CG Accel Lateral'], 
                     c=df['Steering Angle'], cmap='coolwarm', alpha=0.8, 
                     s=15, vmin=-10, vmax=10)
    axes[0, 0].set_title('Lateral G vs Speed', color='white', fontsize=14)
    axes[0, 0].set_xlabel('Speed (km/h)', color='white', fontsize=12)
    axes[0, 0].set_ylabel('Lateral G', color='white', fontsize=12)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Steering Angle (deg)', color='white', fontsize=12)
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
    
    scatter2 = axes[0, 1].scatter(df['Ground Speed'], df['CG Accel Longitudinal'], 
                     c=df['Throttle Pos']-df['Brake Pos'], cmap='coolwarm', 
                     alpha=0.8, s=15, vmin=-100, vmax=100)
    axes[0, 1].set_title('Longitudinal G vs Speed', color='white', fontsize=14)
    axes[0, 1].set_xlabel('Speed (km/h)', color='white', fontsize=12)
    axes[0, 1].set_ylabel('Longitudinal G', color='white', fontsize=12)
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('Throttle-Brake (%)', color='white', fontsize=12)
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')
    
    scatter3 = axes[1, 0].scatter(df['CG Accel Lateral'], df['CG Accel Longitudinal'], 
                     c=df['Ground Speed'], cmap='viridis', alpha=0.8, s=15)
    axes[1, 0].set_title('G-G Diagram', color='white', fontsize=14)
    axes[1, 0].set_xlabel('Lateral G', color='white', fontsize=12)
    axes[1, 0].set_ylabel('Longitudinal G', color='white', fontsize=12)
    cbar3 = plt.colorbar(scatter3, ax=axes[1, 0])
    cbar3.set_label('Speed (km/h)', color='white', fontsize=12)
    cbar3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')
    
    max_g = max(df['CG Accel Lateral'].abs().max(), df['CG Accel Longitudinal'].abs().max()) * 1.1
    circle = plt.Circle((0, 0), 1, fill=False, color='white', linestyle='--', linewidth=1.5)
    axes[1, 0].add_artist(circle)
    axes[1, 0].set_xlim(-max_g, max_g)
    axes[1, 0].set_ylim(-max_g, max_g)
    axes[1, 0].grid(True)

    df.plot(x='Time', y='Chassis Pitch Angle', ax=axes[1, 1], 
            color='#40E0D0', linewidth=2, label='Pitch Angle')
    df.plot(x='Time', y='Chassis Roll Angle', ax=axes[1, 1], 
            color='#FFFF00', linewidth=2, label='Roll Angle')
    axes[1, 1].set_title('Chassis Movements', color='white', fontsize=14)
    axes[1, 1].set_xlabel('Time (s)', color='white', fontsize=12)
    axes[1, 1].set_ylabel('Angle (deg)', color='white', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    
    for ax in axes.flat:
        ax.set_facecolor('#0A0A0A') 
        ax.grid(color='#333333', linestyle='-', linewidth=0.8, alpha=0.8)  
        
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_color('white')
         
        for spine in ax.spines.values():
            spine.set_edgecolor('#FFFFFF')  
            spine.set_linewidth(1.0)
            
        ax.tick_params(colors='white', labelsize=11)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, timestamp_str+'_vehicle_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

def create_telemetry_dashboard(
    df: pd.DataFrame, 
    metadata: dict[str, str], 
    save_dir: Optional[str] = None, 
    auto_scale: bool = True
) -> go.Figure:
    """
    Create an interactive telemetry dashboard using Plotly with improved layout.
    
    Generates a comprehensive interactive dashboard with multiple visualizations:
    - Track map with position and speed visualization
    - G-G diagram showing lateral and longitudinal forces
    - Speed, RPM, and gear visualization versus distance
    - Vehicle dynamics showing roll and pitch angles
    - Driver inputs (throttle, brake, steering) versus distance
    
    Args:
        df (pd.DataFrame): Telemetry data containing all required metrics
        metadata (dict[str, str]): Session metadata including vehicle, track, driver info
        save_dir (Optional[str], optional): Directory to save the HTML dashboard.
            If None, only returns the figure without saving. Defaults to None.
        auto_scale (bool, optional): Whether to use automatic scaling for all plots.
            If False, uses predefined ranges for better comparison. Defaults to True.
            
    Returns:
        go.Figure: Plotly figure object containing the interactive dashboard
    """
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "polar"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"colspan": 2, "type": "xy"}, None], 
        ],
        subplot_titles=(
            "<b>Track Map</b>", "<b>G-G Diagram</b>", 
            "<b>Speed and RPM vs Distance</b>", "<b>Vehicle Dynamics</b>",
            "<b>Driver Inputs vs Distance</b>"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        row_heights=[0.5, 0.5, 0.5]
    )

    vehicle = metadata.get('Vehicle', 'Unknown').replace('ks_', '').replace('_', ' ').title()
    track = metadata.get('Venue', 'Unknown').replace('ks_', '').replace('_', ' ').title()
    driver = metadata.get('Driver', 'Unknown')
    
    max_speed = df['Ground Speed'].max()
    max_rpm = df['Engine RPM'].max()
    lap_distance = df['Distance'].max()
    lap_time = df['Time'].max()
    
    title_text = (f"<b>Telemetry Dashboard</b> - {vehicle} at {track} by {driver}<br>" +
                 f"<span style='font-size:14px'>Distance: {lap_distance:.1f}m | " +
                 f"Lap Time: {lap_time:.2f}s | " +
                 f"Max Speed: {max_speed:.1f}km/h | " +
                 f"Max RPM: {max_rpm:.0f}</span>")
    
    # TRACK MAP (Row 1, Col 1)
    fig.add_trace(
        go.Scatter(
            x=df['Car Coord X'], 
            y=df['Car Coord Y'],
            mode="lines+markers",
            marker=dict(
                size=4,
                color=df['Ground Speed'],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Speed<br>(km/h)",
                    x=0.45,         
                    thickness=10,   
                    len=0.3,
                    y=0.8,          
                ),
                opacity=0.8
            ),
            line=dict(
                width=2,
                color='rgba(255, 255, 255, 0.3)'
            ),
            name="Track Path",
            hovertemplate="X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Speed: %{marker.color:.1f}km/h"
        ),
        row=1, col=1
    )

    start_idx = df['Time'].idxmin()
    fig.add_trace(
        go.Scatter(
            x=[df.loc[start_idx, 'Car Coord X']],
            y=[df.loc[start_idx, 'Car Coord Y']],
            mode="markers",
            marker=dict(
                symbol="star",
                size=12,
                color="red",
                line=dict(width=1, color="white")
            ),
            name="Start/Finish",
            hovertemplate="Start/Finish Line"
        ),
        row=1, col=1
    )

    # G-G DIAGRAM (Row 1, Col 2)
    r = np.sqrt(df['CG Accel Lateral']**2 + df['CG Accel Longitudinal']**2)
    theta = np.arctan2(df['CG Accel Lateral'], df['CG Accel Longitudinal']) * 180 / np.pi
    
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            mode="markers",
            marker=dict(
                size=4,
                color=df['Ground Speed'],
                colorscale="Viridis",
                showscale=False,
                opacity=0.7
            ),
            name="G Forces",
            hovertemplate="Lateral G: %{customdata[0]:.2f}<br>Longitudinal G: %{customdata[1]:.2f}<br>Speed: %{marker.color:.1f}km/h",
            customdata=np.stack((df['CG Accel Lateral'], df['CG Accel Longitudinal']), axis=1)
        ),
        row=1, col=2
    )

    # Add 1G reference circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    r_circle = np.ones(100)
    fig.add_trace(
        go.Scatterpolar(
            r=r_circle,
            theta=theta_circle*180/np.pi,
            mode="lines",
            line=dict(color="white", width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=2
    )
    
    # SPEED/RPM PLOT (Row 2, Col 1)
    fig.add_trace(
        go.Scatter(
            x=df['Distance'], 
            y=df['Ground Speed'],
            name="Speed",
            line=dict(color="#00FFFF", width=2),
            hovertemplate="Distance: %{x:.1f}m<br>Speed: %{y:.1f}km/h"
        ),
        row=2, col=1
    )
    
    # Add RPM scaled down by 10 for better visualization
    fig.add_trace(
        go.Scatter(
            x=df['Distance'], 
            y=df['Engine RPM'] / 10,
            name="RPM (x10)",
            line=dict(color="#FFA500", width=2),
            hovertemplate="Distance: %{x:.1f}m<br>RPM: %{y:.1f}0",
            yaxis="y3"
        ),
        row=2, col=1
    )

    # Add gear display if available
    if 'Gear' in df.columns:
        gear_height = df['Ground Speed'].max() * 0.15
        
        fig.add_trace(
            go.Scatter(
                x=[df['Distance'].min(), df['Distance'].max()],
                y=[0, 0],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(50, 50, 50, 0.2)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Distance'],
                y=df['Gear'] * gear_height,
                name="Gear",
                line=dict(color="#50FF78", width=2, shape='hv'),
                hovertemplate="Distance: %{x:.1f}m<br>Gear: %{customdata:.0f}",
                customdata=df['Gear'],
                fill='tozeroy',
                fillcolor='rgba(80, 255, 120, 0.1)'
            ),
            row=2, col=1
        )
        
        # Add gear number labels at gear changes
        gear_changes = df[['Distance', 'Gear']].copy()
        gear_changes['next_gear'] = gear_changes['Gear'].shift(-1)
        gear_changes['gear_change'] = gear_changes['Gear'] != gear_changes['next_gear']
        gear_changes = gear_changes[gear_changes['gear_change']]
        
        for i, row in gear_changes.iterrows():
            if i > 0 and row['Gear'] > 0:
                prev_change = gear_changes.iloc[gear_changes.index.get_loc(i)-1]
                mid_x = (prev_change['Distance'] + row['Distance']) / 2
                
                fig.add_annotation(
                    x=mid_x,
                    y=row['Gear'] * gear_height * 0.5,
                    text=f"{int(row['Gear'])}",
                    showarrow=False,
                    font=dict(color="white", size=10, family="Arial Black"),
                    row=2, col=1
                )

    # VEHICLE DYNAMICS (Row 2, Col 2)
    fig.add_trace(
        go.Scatter(
            x=df['Time'], 
            y=df['Chassis Roll Angle'],
            name="Roll Angle",
            line=dict(color="#FFFF00", width=2),
            hovertemplate="Time: %{x:.2f}s<br>Roll: %{y:.2f}°"
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Time'], 
            y=df['Chassis Pitch Angle'],
            name="Pitch Angle",
            line=dict(color="#40E0D0", width=2),
            hovertemplate="Time: %{x:.2f}s<br>Pitch: %{y:.2f}°"
        ),
        row=2, col=2
    )

    # DRIVER INPUTS (Row 3, Col 1 - full width)
    fig.add_trace(
        go.Scatter(
            x=df['Distance'], 
            y=df['Throttle Pos'],
            name="Throttle",
            line=dict(color="#CCFF00", width=2),
            fill='tozeroy',
            fillcolor='rgba(204, 255, 0, 0.2)',
            hovertemplate="Distance: %{x:.1f}m<br>Throttle: %{y:.1f}%"
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Distance'], 
            y=df['Brake Pos'],
            name="Brake",
            line=dict(color="#F31D1D", width=2),
            fill='tozeroy',
            fillcolor='rgba(243, 29, 29, 0.2)',
            hovertemplate="Distance: %{x:.1f}m<br>Brake: %{y:.1f}%"
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Distance'], 
            y=df['Steering Angle'],
            name="Steering",
            line=dict(color="#FFFFFF", width=2),
            yaxis="y6",
            hovertemplate="Distance: %{x:.1f}m<br>Steering: %{y:.1f}°"
        ),
        row=3, col=1
    )
    
    # Calculate ranges for each plot
    x_min, x_max = df['Car Coord X'].min(), df['Car Coord X'].max()
    y_min, y_max = df['Car Coord Y'].min(), df['Car Coord Y'].max()
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range) * 1.1
    
    max_g = max(df['CG Accel Lateral'].abs().max(), df['CG Accel Longitudinal'].abs().max()) * 1.2
    chassis_min = min(df['Chassis Roll Angle'].min(), df['Chassis Pitch Angle'].min()) * 1.2
    chassis_max = max(df['Chassis Roll Angle'].max(), df['Chassis Pitch Angle'].max()) * 1.2
    steering_range = max(abs(df['Steering Angle'].min()), abs(df['Steering Angle'].max())) * 1.2
    
    # Create axis configuration dictionaries
    xaxis1_config = {
        "title": "X Position (m)",
        "scaleanchor": "y",  # Maintain equal aspect ratio
        "scaleratio": 1,
    }
    
    yaxis1_config = {
        "title": "Y Position (m)",
    }
    
    # Only add manual ranges if auto_scale is False
    if not auto_scale:
        xaxis1_config["range"] = [x_center - max_range/2, x_center + max_range/2]
        yaxis1_config["range"] = [y_center - max_range/2, y_center + max_range/2]
    
    # Speed and RPM axes
    yaxis2_config = {
        "title": dict(text="Speed (km/h)", font=dict(color="#00FFFF")),
        "titlefont": dict(color="#00FFFF"),
        "tickfont": dict(color="#00FFFF"),
        "side": "left",
        "showgrid": True,
    }
    
    yaxis3_config = {
        "title": dict(text="RPM (x10)", font=dict(color="#FFA500")),
        "titlefont": dict(color="#FFA500"),
        "tickfont": dict(color="#FFA500"),
        "side": "right",
        "overlaying": "y2",
        "showgrid": False,
    }
    
    if not auto_scale:
        yaxis2_config["range"] = [0, df['Ground Speed'].max() * 1.1]
        yaxis3_config["range"] = [0, df['Engine RPM'].max() / 10 * 1.1]
    
    # Vehicle dynamics axes
    yaxis4_config = {
        "title": dict(text="Angle (°)", font=dict(color="white")),
        "tickmode": "auto",
        "nticks": 10
    }
    
    if not auto_scale:
        dynamics_range = max(abs(chassis_min), abs(chassis_max))
        yaxis4_config["range"] = [-dynamics_range, dynamics_range]
    
    # Driver inputs axes
    yaxis5_config = {
        "title": dict(text="Pedal Position (%)", font=dict(color="#CCFF00")),
        "titlefont": dict(color="#CCFF00"),
        "tickfont": dict(color="#CCFF00"),
        "side": "left",
        "showgrid": True,
        "domain": [0.15, 0.85]
    }
    
    # Throttle/brake are percentages, so fixed range makes sense even in auto mode
    yaxis5_config["range"] = [0, 105]
    
    yaxis6_config = {
        "title": dict(text="Steering Angle (°)", font=dict(color="#FFFFFF")),
        "titlefont": dict(color="#FFFFFF"),
        "tickfont": dict(color="#FFFFFF"),
        "anchor": "x4",
        "overlaying": "y5",
        "side": "right",
        "showgrid": False,
    }
    
    if not auto_scale:
        yaxis6_config["range"] = [-steering_range, steering_range]
    
    # G-G diagram config
    polar_config = {
        "radialaxis": dict(
            visible=True,
            tickvals=[0.5, 1.0, 1.5, 2.0],
            ticktext=["0.5G", "1.0G", "1.5G", "2.0G"],
        ),
        "angularaxis": dict(
            rotation=90,
            direction="clockwise",
            tickvals=[0, 90, 180, 270],
            ticktext=["Accel", "Right", "Brake", "Left"],
        ),
        "bgcolor": "rgba(20, 20, 20, 0.8)",
    }
    
    if not auto_scale:
        polar_config["radialaxis"]["range"] = [0, max_g]
    
    # Update layout with axis configurations
    fig.update_layout(
        xaxis1=xaxis1_config,
        yaxis1=yaxis1_config,
        xaxis2=dict(title="Distance (m)"),
        yaxis2=yaxis2_config,
        yaxis3=yaxis3_config,
        xaxis3=dict(title="Time (s)"),
        yaxis4=yaxis4_config,
        xaxis4=dict(title="Distance (m)"),
        yaxis5=yaxis5_config,
        yaxis6=yaxis6_config,
        polar=polar_config
    )
    
    # Global layout settings
    fig.update_layout(
        height=2000,
        width=1890,
        template="plotly_dark",
        paper_bgcolor='#0A0A0A',
        plot_bgcolor='#0A0A0A',
        title=dict(
            text=title_text,
            font=dict(size=20),
            x=0.5,
            xanchor="center"
        ),
        font=dict(
            family="Arial, sans-serif",
            color='white',
            size=13
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            bgcolor="rgba(10,10,10,0.7)",
            bordercolor="rgba(60,60,60,0.8)",
            borderwidth=1
        ),
        margin=dict(l=50, r=50, b=50, t=120, pad=4),
        hovermode="closest",
    )
    
    # Grid styling for all subplots
    fig.update_xaxes(
        showgrid=True, 
        gridcolor='#333333', 
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#777777', 
        zerolinewidth=1
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='#333333', 
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#777777', 
        zerolinewidth=1
    )
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.add_annotation(
        text=f"Generated: {timestamp}",
        x=1,
        y=0,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            size=10,
            color="rgba(200, 200, 200, 0.5)"
        ),
        align="right"
    )
    
    # Add autoscale note if enabled
    if auto_scale:
        fig.add_annotation(
            text="Auto-scaling enabled",
            x=0,
            y=0,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=10,
                color="rgba(200, 200, 200, 0.5)"
            ),
            align="left"
        )
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.write_html(
            os.path.join(save_dir, timestamp_str+"_telemetry_dashboard.html"),
            include_plotlyjs="cdn",
            full_html=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'responsive': True
            }
        )
    
    return fig

df, metadata = load_motec_csv('test_trial_lap.csv')
plot_time_series(df, save_dir='plots')
plot_track_map(df, save_dir='plots')
plot_driver_inputs(df, save_dir='plots')
plot_vehicle_dynamics(df, save_dir='plots')
dashboard_fig = create_telemetry_dashboard(df, metadata, save_dir='plots')