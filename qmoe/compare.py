import matplotlib.pyplot as plt
import numpy as np

# 数据
data_sizes = ['15.6MB', '15.6MB', '31.3MB', '93.8MB', '125MB']
group1_bandwidth = [2180.81 / 4, 2155.84 / 4, 2680.70 / 4, 3167.81  / 4 , 3151.05 / 4]
group2_bandwidth = [2679.0464 / 4, 2164.0 / 4, 3856.75 / 4, 4705.85 / 4, 4835.43 / 4 ]

# bandwidth_gb_s=2679.0464038977866 GB/s, 0.0312652587890625 GB, 0.011670293856640252 ms
# bandwidth_gb_s=2164.0516156333233 GB/s, 0.015636444091796875 GB, 0.007225541192658094 ms
# bandwidth_gb_s=3015.889139468831 GB/s, 0.06252288818359375 GB, 0.020731162616476513 ms
# bandwidth_gb_s=3139.9031152768484 GB/s, 0.093780517578125 GB, 0.029867328428653212 ms
# bandwidth_gb_s=3148.9920091536555 GB/s, 0.125030517578125 GB, 0.03970493326584498 ms
# 设置图形
plt.figure(figsize=(12, 8))

# 设置柱状图位置
x = np.arange(len(data_sizes))
width = 0.35

# 绘制柱状图
bars1 = plt.bar(x - width/2, group1_bandwidth, width, label='triton', color='skyblue', edgecolor='black', alpha=0.8)
bars2 = plt.bar(x + width/2, group2_bandwidth, width, label='triton inline ptx', color='lightcoral', edgecolor='black', alpha=0.8)

# 添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)

# 设置图形属性
plt.xlabel('data size', fontsize=12, fontweight='bold')
plt.ylabel('bandwidth (GB/s)', fontsize=12, fontweight='bold')
plt.title('compare', fontsize=14, fontweight='bold')
plt.xticks(x, data_sizes)
plt.legend()

# 添加网格
plt.grid(True, axis='y', alpha=0.3, linestyle='--')

# 设置y轴范围
plt.ylim(0, max(max(group1_bandwidth), max(group2_bandwidth)) * 1.1)

# 保存为PNG文件
plt.tight_layout()
plt.savefig('bandwidth_comparison.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭当前图形，为下一个图形做准备

# 计算性能提升百分比
improvement = [(g2 - g1) / g1 * 100 for g1, g2 in zip(group1_bandwidth, group2_bandwidth)]

# 绘制性能提升图
plt.figure(figsize=(10, 6))
bars_improve = plt.bar(data_sizes, improvement, color=['gold' if x > 0 else 'lightblue' for x in improvement], 
                       edgecolor='black', alpha=0.8)

# 添加数值标签
for i, bar in enumerate(bars_improve):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1 if height > 0 else height - 3,
            f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')

plt.xlabel('数据大小', fontsize=12, fontweight='bold')
plt.ylabel('性能提升 (%)', fontsize=12, fontweight='bold')
plt.title('第二组相对于第一组的性能提升百分比', fontsize=14, fontweight='bold')
plt.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.8)

# 保存为PNG文件
plt.tight_layout()
plt.savefig('performance_improvement.png', dpi=300, bbox_inches='tight')
plt.close()

# 打印详细数据对比
print("详细数据对比:")
print("数据大小\t第一组(GB/s)\t第二组(GB/s)\t提升百分比")
print("-" * 55)
for i in range(len(data_sizes)):
    print(f"{data_sizes[i]}\t{group1_bandwidth[i]:.2f}\t\t{group2_bandwidth[i]:.2f}\t\t{improvement[i]:.2f}%")

print("\n图表已保存为:")
print("- bandwidth_comparison.png")
print("- performance_improvement.png")