import numpy as np
import matplotlib.pyplot as plt
import os

# Create grid of Precision and Recall
p = np.linspace(0.01, 1.0, 500)
r = np.linspace(0.01, 1.0, 500)
P, R = np.meshgrid(p, r)

# Calculate F1 Score
F1 = 2 * (P * R) / (P + R)

# Set style
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
ax.grid(True, linestyle='--', alpha=0.5)

# Plot contours
contour = ax.contourf(P, R, F1, levels=20, cmap='viridis')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('F1-Score', fontsize=11, fontweight='bold', labelpad=10)
cbar.ax.tick_params(labelsize=9)

# Add contour lines with labels
lines = ax.contour(P, R, F1, levels=10, colors='white', alpha=0.6, linewidths=1.0)
ax.clabel(lines, inline=True, fmt='%.1f', fontsize=9, colors='white')

# Labels and title
ax.set_xlabel('Precision (Độ chính xác)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('Recall (Độ thu hồi)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_title('Sự phụ thuộc của F1-Score vào Precision và Recall', fontsize=12, fontweight='bold', pad=15)

# Equal F1 diagonal
ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1.5, label='Precision = Recall (F1 = P = R)')
ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_aspect('equal')

plt.tight_layout()

# Save the plot
output_path = os.path.join('figures', 'f1_score_dependency.png')
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Successfully generated {output_path}")
