import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# 创建数据
data = [
    # Group 1 数据
    {'method': 'torch-f16', 'm': 2048, 'n': 4096, 'time': 0.020688, 'group': 'Group 1'},
    {'method': 'mutis', 'm': 2048, 'n': 4096, 'time': 0.011040, 'group': 'Group 1'},
    {'method': 'Qtriton', 'm': 2048, 'n': 4096, 'time': 0.0080533301, 'group': 'Group 1'},
    
    {'method': 'torch-f16', 'm': 4096, 'n': 8192, 'time': 0.055168, 'group': 'Group 1'},
    {'method': 'mutis', 'm': 4096, 'n': 8192, 'time': 0.018576, 'group': 'Group 1'},
    {'method': 'Qtriton', 'm': 4096, 'n': 8192, 'time': 0.0169639729, 'group': 'Group 1'},
    
    {'method': 'torch-f16', 'm': 12288, 'n': 4096, 'time': 0.078512, 'group': 'Group 1'},
    {'method': 'mutis', 'm': 12288, 'n': 4096, 'time': 0.023776, 'group': 'Group 1'},
    {'method': 'Qtriton', 'm': 12288, 'n': 4096, 'time': 0.021730357072, 'group': 'Group 1'},
    
    {'method': 'torch-f16', 'm': 8192, 'n': 8192, 'time': 0.096480, 'group': 'Group 1'},
    {'method': 'mutis', 'm': 8192, 'n': 8192, 'time': 0.028496, 'group': 'Group 1'},
    {'method': 'Qtriton', 'm': 8192, 'n': 8192, 'time': 0.0267648, 'group': 'Group 1'},
    
    # Group 2 数据
    {'method': 'torch-f16', 'm': 4096, 'n': 4096, 'time': 0.036864, 'group': 'Group 2'},
    {'method': 'mutis', 'm': 4096, 'n': 4096, 'time': 0.014368, 'group': 'Group 2'},
    {'method': 'Qtriton', 'm': 4096, 'n': 4096, 'time': 0.010842526149, 'group': 'Group 2'},
    
    {'method': 'torch-f16', 'm': 8192, 'n': 8192, 'time': 0.096480, 'group': 'Group 2'},
    {'method': 'mutis', 'm': 8192, 'n': 8192, 'time': 0.028496, 'group': 'Group 2'},
    {'method': 'Qtriton', 'm': 8192, 'n': 8192, 'time': 0.02672976785, 'group': 'Group 2'},
    
    {'method': 'torch-f16', 'm': 8192, 'n': 57344, 'time': 0.507120, 'group': 'Group 2'},
    {'method': 'mutis', 'm': 8192, 'n': 57344, 'time': 0.143616, 'group': 'Group 2'},
    {'method': 'Qtriton', 'm': 8192, 'n': 57344, 'time': 0.1423156176, 'group': 'Group 2'},
    
    {'method': 'torch-f16', 'm': 28672, 'n': 8192, 'time': 0.264976, 'group': 'Group 2'},
    {'method': 'mutis', 'm': 28672, 'n': 8192, 'time': 0.078656, 'group': 'Group 2'},
    {'method': 'Qtriton', 'm': 28672, 'n': 8192, 'time': 0.07595430027276, 'group': 'Group 2'}
]

df = pd.DataFrame(data)

# 创建矩阵大小标签
df['matrix_size'] = df.apply(lambda x: f"{x['m']}×{x['n']}", axis=1)
df['elements'] = df['m'] * df['n']

# 计算加速比（修复错误）
def calculate_speedup(df):
    # 创建一个字典来存储每个矩阵大小的基准时间（torch-f16）
    base_times = {}
    for idx, row in df.iterrows():
        if row['method'] == 'torch-f16':
            key = (row['m'], row['n'])
            base_times[key] = row['time']
    
    # 计算加速比
    speedups = []
    for idx, row in df.iterrows():
        key = (row['m'], row['n'])
        if row['method'] == 'torch-f16':
            speedups.append(1.0)
        elif key in base_times and base_times[key] > 0:
            speedup = base_times[key] / row['time']
            speedups.append(speedup)
        else:
            speedups.append(np.nan)
    
    return speedups

df['speedup'] = calculate_speedup(df)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NVIDIA-H100-PCIe 矩阵计算性能对比\n(torch-f16 vs mutis vs Qtriton)', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. 执行时间对比（按方法分组）
ax1 = axes[0, 0]
matrix_sizes = df['matrix_size'].unique()
methods = ['torch-f16', 'mutis', 'Qtriton']

# 准备数据
time_data = {}
for method in methods:
    method_df = df[df['method'] == method]
    times = []
    for size in matrix_sizes:
        size_time = method_df[method_df['matrix_size'] == size]['time']
        times.append(size_time.values[0] if len(size_time) > 0 else np.nan)
    time_data[method] = times

x = np.arange(len(matrix_sizes))
width = 0.25

for i, method in enumerate(methods):
    offset = width * (i - 1)  # 居中显示
    bars = ax1.bar(x + offset, time_data[method], width, label=method, alpha=0.8)
    
    # 添加数值标签（如果时间不太小）
    for bar, val in zip(bars, time_data[method]):
        if not np.isnan(val) and val > 0.01:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('矩阵大小 (m×n)', fontsize=12)
ax1.set_ylabel('执行时间 (秒)', fontsize=12)
ax1.set_title('执行时间对比 (柱状图)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(matrix_sizes, rotation=45, ha='right')
ax1.legend(title='计算方法')
ax1.grid(True, alpha=0.3, axis='y')

# 2. 加速比对比
ax2 = axes[0, 1]
int4_methods = ['mutis', 'Qtriton']
speedup_data = {method: [] for method in int4_methods}

for method in int4_methods:
    for size in matrix_sizes:
        speedup = df[(df['method'] == method) & (df['matrix_size'] == size)]['speedup']
        speedup_data[method].append(speedup.values[0] if len(speedup) > 0 else np.nan)

width = 0.35
for i, method in enumerate(int4_methods):
    offset = width * (i - 0.5)
    valid_data = [x if not np.isnan(x) else 0 for x in speedup_data[method]]
    bars = ax2.bar(x + offset, valid_data, width, label=method, alpha=0.8)
    
    # 添加数值标签
    for bar, val in zip(bars, speedup_data[method]):
        if not np.isnan(val):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=9)

ax2.set_xlabel('矩阵大小 (m×n)', fontsize=12)
ax2.set_ylabel('加速比 (相对于 torch-f16)', fontsize=12)
ax2.set_title('加速比对比', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(matrix_sizes, rotation=45, ha='right')
ax2.legend(title='INT4 方法')
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='基准线 (torch-f16)')
ax2.grid(True, alpha=0.3, axis='y')

# 3. 热力图
ax3 = axes[1, 0]
# 准备数据用于热力图
heatmap_data = []
for size in matrix_sizes:
    row = []
    for method in methods:
        time_val = df[(df['matrix_size'] == size) & (df['method'] == method)]['time']
        row.append(time_val.values[0] if len(time_val) > 0 else np.nan)
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)
im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

# 设置坐标轴标签
ax3.set_xticks(np.arange(len(methods)))
ax3.set_xticklabels(methods)
ax3.set_yticks(np.arange(len(matrix_sizes)))
ax3.set_yticklabels(matrix_sizes)
ax3.set_title('执行时间热力图 (秒)', fontsize=14, fontweight='bold')

# 添加数值到热力图
for i in range(len(matrix_sizes)):
    for j in range(len(methods)):
        if not np.isnan(heatmap_data[i, j]):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('执行时间 (秒)', rotation=270, labelpad=15)

# 4. 矩阵规模 vs 执行时间散点图
ax4 = axes[1, 1]
colors = {'torch-f16': 'red', 'mutis': 'blue', 'Qtriton': 'green'}
markers = {'torch-f16': 'o', 'mutis': 's', 'Qtriton': '^'}

for method in methods:
    method_df = df[df['method'] == method]
    ax4.scatter(method_df['elements'], method_df['time'], 
                c=colors[method], marker=markers[method], s=100,
                label=method, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # 添加趋势线
    if len(method_df) > 1:
        x_vals = method_df['elements'].values
        y_vals = method_df['time'].values
        # 使用对数坐标进行线性拟合
        log_x = np.log10(x_vals)
        log_y = np.log10(y_vals)
        coeffs = np.polyfit(log_x, log_y, 1)
        trend_x = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 100)
        trend_y = 10**(coeffs[0] * np.log10(trend_x) + coeffs[1])
        ax4.plot(trend_x, trend_y, color=colors[method], linestyle='--', alpha=0.5)

ax4.set_xlabel('矩阵元素数量 (m×n)', fontsize=12)
ax4.set_ylabel('执行时间 (秒)', fontsize=12)
ax4.set_title('矩阵规模 vs 执行时间 (对数坐标)', fontsize=14, fontweight='bold')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, which='both')
ax4.legend(title='计算方法')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存为PNG文件
plt.savefig('nvidia_h100_performance_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')

print("图表已保存为 'nvidia_h100_performance_comparison.png'")

# 显示图表
plt.show()

# 打印汇总统计
print("\n" + "="*80)
print("性能汇总统计:")
print("="*80)

# 按方法统计
for method in methods:
    method_df = df[df['method'] == method]
    print(f"\n{method}:")
    print(f"  平均执行时间: {method_df['time'].mean():.4f} 秒")
    print(f"  最快执行时间: {method_df['time'].min():.4f} 秒")
    print(f"  最慢执行时间: {method_df['time'].max():.4f} 秒")

# 计算平均加速比
print("\n平均加速比 (相对于 torch-f16):")
for method in ['mutis', 'Qtriton']:
    speedups = df[df['method'] == method]['speedup'].dropna()
    if len(speedups) > 0:
        print(f"  {method}: {speedups.mean():.2f}x 加速")

# 显示数据表格
print("\n详细数据:")
print(df[['matrix_size', 'method', 'time', 'speedup']].to_string(index=False))