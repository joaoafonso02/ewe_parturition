import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Enable interactive mode
plt.ion()

# Read data
df = pd.read_csv('model_comparison_results.csv')

# Create grid
delta_values = sorted(df['Delta_Time'].unique())
rate_values = sorted(df['Sample_Rate'].unique())
delta_grid, rate_grid = np.meshgrid(delta_values, rate_values)

# Create separate figures
models = ['RF', 'DT', 'KNN']
titles = ['Random Forest', 'Decision Tree', 'K-Nearest Neighbors']
colors = ['Blues', 'Greens', 'Reds']
figs = []

for idx, (model, title, cmap) in enumerate(zip(models, titles, colors)):
    model_data = df[df['Model'] == model]
    
    # Create MCC matrix
    mcc_matrix = np.zeros((len(rate_values), len(delta_values)))
    for i, rate in enumerate(rate_values):
        for j, delta in enumerate(delta_values):
            mcc = model_data[(model_data['Sample_Rate'] == rate) & 
                           (model_data['Delta_Time'] == delta)]['MCC'].values
            mcc_matrix[i, j] = mcc[0] if len(mcc) > 0 else np.nan
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(delta_grid, rate_grid, mcc_matrix, 
                          cmap=cmap, edgecolor='none')
    
    # Customize plot
    ax.set_xlabel('Delta Time (minutes)')
    ax.set_ylabel('Sample Rate (Hz)')
    ax.set_zlabel('MCC')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Store figure reference
    figs.append(fig)

# Function to update views
def on_key(event):
    if event.key == 'left':
        for ax in [f.gca() for f in figs]:
            ax.view_init(elev=ax.elev, azim=ax.azim - 10)
    elif event.key == 'right':
        for ax in [f.gca() for f in figs]:
            ax.view_init(elev=ax.elev, azim=ax.azim + 10)
    elif event.key == 'up':
        for ax in [f.gca() for f in figs]:
            ax.view_init(elev=ax.elev + 10, azim=ax.azim)
    elif event.key == 'down':
        for ax in [f.gca() for f in figs]:
            ax.view_init(elev=ax.elev - 10, azim=ax.azim)
    plt.draw()

# Connect key event handler
for fig in figs:
    fig.canvas.mpl_connect('key_press_event', on_key)

print("Use arrow keys to rotate the plots")
plt.show(block=True)
