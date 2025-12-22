import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = {
    'device': ['NVIDIA-GeForce-RTX-4090'] * 16,
    'runner': [
        'torch-f16', 'marlin', 'mutis', 'Qtriton',
        'torch-f16', 'marlin', 'mutis', 'Qtriton',
        'torch-f16', 'marlin', 'mutis', 'Qtriton',
        'torch-f16', 'marlin', 'mutis', 'QTriton'
    ],
    'a_dtype': ['float16'] * 16,
    'b_dtype': ['float16', 'int4b', 'int4b', 'int4b'] * 4,
    'group_size': [-1, 128, 128, -1, -1, 128, 128, -1, -1, 128, 128, 128, -1, 128, 128, 128],
    'm': [1] * 16,
    'k': [4096, 4096, 4096, 4096, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 28672, 28672, 28672, 28672],
    'n': [4096, 4096, 4096, 4096, 8192, 8192, 8192, 8192, 57344, 57344, 57344, 57344, 8192, 8192, 8192, 8192],
    'latency': [
        0.051200, 0.021520, 0.018432, 0.00941298304019,
        0.191456, 0.049152, 0.047104, 0.037080677498515784,
        1.076224, 0.268288, 0.264192, 0.2656944067735248,
        0.544768, 0.139264, 0.137152, 0.13785157000236903
    ]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 创建矩阵大小的标签
df['matrix_size'] = df.apply(lambda row: f"({row['m']},{row['k']},{row['n']})", axis=1)
df['operation_size'] = df.apply(lambda row: f"{row['k']}×{row['n']}", axis=1)

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Comparison of Different Runners on NVIDIA RTX 4090', fontsize=16, fontweight='bold')

# 定义颜色
colors = {
    'torch-f16': '#2E86AB',
    'marlin': '#A23B72',
    'mutis': '#F18F01',
    'Qtriton': '#73AB84',
    'QTriton': '#73AB84'
}

# 按矩阵大小分组
matrix_groups = [
    (1, 4096, 4096),
    (1, 8192, 8192),
    (1, 8192, 57344),
    (1, 28672, 8192)
]

titles = [
    'Small Matrix (1,4096,4096)',
    'Medium Matrix (1,8192,8192)',
    'Tall Matrix (1,8192,57344)',
    'Wide Matrix (1,28672,8192)'
]

for idx, (m, k, n) in enumerate(matrix_groups):
    ax = axes[idx // 2, idx % 2]
    
    # 筛选当前矩阵大小的数据
    group_data = df[(df['m'] == m) & (df['k'] == k) & (df['n'] == n)]
    
    # 提取runner和latency
    runners = group_data['runner'].tolist()
    latencies = group_data['latency'].tolist()
    
    # 创建柱状图
    bars = ax.bar(runners, latencies, color=[colors[r] for r in runners])
    
    # 添加数值标签
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 设置图表属性
    ax.set_title(titles[idx], fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=10)
    ax.set_xlabel('Runner', fontsize=10)
    ax.set_ylim(0, max(latencies) * 1.2)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 旋转x轴标签
    ax.set_xticklabels(runners, rotation=45, ha='right')

# 调整布局
plt.tight_layout()

# 保存为PNG
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为 'performance_comparison.png'")

# 显示图表
plt.show()

# 创建汇总对比图表
fig2, ax2 = plt.subplots(figsize=(14, 8))

# 为每个runner准备数据
runners = ['torch-f16', 'marlin', 'mutis', 'Qtriton']
x = np.arange(len(matrix_groups))
width = 0.2

for i, runner in enumerate(runners):
    runner_data = []
    for m, k, n in matrix_groups:
        data_point = df[(df['runner'] == runner) & 
                       (df['m'] == m) & (df['k'] == k) & (df['n'] == n)]
        if not data_point.empty:
            runner_data.append(data_point['latency'].values[0])
        else:
            runner_data.append(0)
    
    offset = (i - 1.5) * width
    bars = ax2.bar(x + offset, runner_data, width, label=runner, color=colors[runner])
    
    # 添加数值标签
    for bar, value in zip(bars, runner_data):
        if value > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# 设置汇总图表属性
ax2.set_xlabel('Matrix Size (m,k,n)', fontsize=12)
ax2.set_ylabel('Latency (ms)', fontsize=12)
ax2.set_title('Performance Comparison Across Different Matrix Sizes', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'({m},{k},{n})' for m, k, n in matrix_groups], rotation=45, ha='right')
ax2.legend(title='Runner')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
print("汇总图表已保存为 'performance_summary.png'")
plt.show()

# 输出数据摘要
print("\n性能数据摘要:")
print("=" * 80)
for m, k, n in matrix_groups:
    print(f"\n矩阵大小: ({m}, {k}, {n})")
    group_data = df[(df['m'] == m) & (df['k'] == k) & (df['n'] == n)]
    fastest = group_data.loc[group_data['latency'].idxmin()]
    print(f"  最快: {fastest['runner']} - {fastest['latency']:.6f} ms")
    
    # 计算速度提升
    torch_f16 = group_data[group_data['runner'] == 'torch-f16']['latency'].values[0]
    for runner in ['marlin', 'mutis', 'Qtriton', 'QTriton']:
        runner_data = group_data[group_data['runner'] == runner]
        if not runner_data.empty:
            latency = runner_data['latency'].values[0]
            speedup = torch_f16 / latency
            print(f"  {runner}: {latency:.6f} ms (相比torch-f16: {speedup:.2f}x)")
