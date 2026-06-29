import numpy as np
import matplotlib.pyplot as plt
import os

# Set output directory
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define SiLU
def silu(x):
    return x * sigmoid(x)

# Define ReLU
def relu(x):
    return np.maximum(0, x)

# Set common font configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# ----------------- Plot 1: SiLU Activation -----------------
x = np.linspace(-5, 5, 500)
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
ax.grid(True, linestyle='--', alpha=0.5)

ax.plot(x, silu(x), label='SiLU (Swish)', color='#1f77b4', linewidth=2.5)
ax.plot(x, relu(x), label='ReLU', color='#2ca02c', linestyle='--', linewidth=1.5)
ax.plot(x, sigmoid(x), label='Sigmoid', color='#7f7f7f', linestyle=':', linewidth=1.5)

# Axis lines
ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
ax.axvline(0, color='black', linewidth=0.8, alpha=0.7)

ax.set_title('So sánh hàm kích hoạt SiLU, ReLU và Sigmoid', fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel('Giá trị đầu vào (x)', fontsize=9)
ax.set_ylabel('Giá trị đầu ra (y)', fontsize=9)
ax.set_xlim(-5, 5)
ax.set_ylim(-1.0, 5)
ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'silu_activation.png'), bbox_inches='tight', dpi=300)
plt.close()

# ----------------- Plot 2: Batch Normalization -----------------
# Normal distribution PDF
def norm_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

x_bn = np.linspace(-6, 8, 1000)
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
ax.grid(True, linestyle='--', alpha=0.5)

# Original distribution: mean = 2.0, std = 1.5
y_orig = norm_pdf(x_bn, 2.0, 1.5)
# Normalized distribution (zero mean, unit var): mean = 0.0, std = 1.0
y_norm = norm_pdf(x_bn, 0.0, 1.0)
# Scaled and shifted distribution: gamma = 1.2, beta = 1.0 (mean = 1.0, std = 1.2)
y_scaled = norm_pdf(x_bn, 1.0, 1.2)

ax.plot(x_bn, y_orig, label=r'Phân phối ban đầu $x$ ($\mu=2.0, \sigma=1.5$)', color='#d62728', linewidth=2)
ax.plot(x_bn, y_norm, label=r'Chuẩn hóa $\hat{x}$ ($\mu=0.0, \sigma=1.0$)', color='#1f77b4', linewidth=2, linestyle='--')
ax.plot(x_bn, y_scaled, label=r'Sau tỷ lệ/dịch chuyển $y$ ($\beta=1.0, \gamma=1.2$)', color='#2ca02c', linewidth=2, linestyle='-.')

ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
ax.axvline(0, color='black', linewidth=0.8, alpha=0.7)

ax.set_title('Minh họa tác động của Batch Normalization', fontsize=11, fontweight='bold', pad=12)
ax.set_xlabel('Giá trị biến kích hoạt', fontsize=9)
ax.set_ylabel('Mật độ xác suất (PDF)', fontsize=9)
ax.set_xlim(-5, 7)
ax.set_ylim(0, 0.5)
ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'batch_norm_distribution.png'), bbox_inches='tight', dpi=300)
plt.close()

print("Successfully generated silu_activation.png and batch_norm_distribution.png")
