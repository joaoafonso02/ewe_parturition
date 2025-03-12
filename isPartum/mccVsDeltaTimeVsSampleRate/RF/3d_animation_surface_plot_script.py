import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file that contains the data.
df = pd.read_csv('mcc_surface_data.csv')

# Pivot the DataFrame so that:
# X: Delta_Time (minutes), Y: Sample_Rate (Hz), Z: MCC.
pivot_df = df.pivot(index='Sample_Rate', columns='Delta_Time', values='MCC')

# Create meshgrid
X = np.array(pivot_df.columns, dtype=float)
Y = np.array(pivot_df.index, dtype=float)
X, Y = np.meshgrid(X, Y)
Z = pivot_df.values

# Plot the 3D surface (interactive; rotate with mouse)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('Delta Time (minutes)')
ax.set_ylabel('Sample Rate (Hz)')
ax.set_zlabel('MCC')
ax.set_title('MCC 3D Visualization')
fig.colorbar(surf, shrink=0.5, aspect=5)

# Find the top 3 MCC values
flattened_indices = np.dstack(np.unravel_index(np.argsort(Z, axis=None)[-3:], Z.shape))[0]  # Get top 3 indices
top_mcc_values = [Z[i, j] for i, j in flattened_indices]
top_points = [(X[i, j], Y[i, j], Z[i, j]) for i, j in flattened_indices]

# Colors for the three points
colors = ['red', 'blue', 'orange']

# Scatter plot for the top 3 points and format legend entries
legend_entries = []
for idx, (delta, rate, mcc) in enumerate(top_points):
    ax.scatter(delta, rate, mcc, color=colors[idx], s=100, depthshade=True)
    legend_entries.append(f"MCC = {mcc:.4f}, Delta Time = {int(delta)} min, Sample Rate = {rate:.1f} Hz")

# Add legend with MCC, Delta Time, and Sample Rate
ax.legend(legend_entries, loc='upper left')

plt.show()

