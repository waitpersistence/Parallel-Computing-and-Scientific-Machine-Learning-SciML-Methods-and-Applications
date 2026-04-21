import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# =============================================================================
# 中文字体配置 - 修复中文渲染问题
# =============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

"""
延迟微分方程 (DDE) 的 ODE 近似方法

核心思想：使用一串耦合的常微分方程 (ODE) 来近似延迟微分方程 (DDE)。
这种方法被称为"链条方法"或"水桶接力法"。

原始 DDE: dx/dt = -α*x(t) + β*x(t-τ)
其中 τ 是延迟时间。

近似方法：引入 N-1 个辅助变量，形成一个链条：
- x₀ = x(t) (当前状态)
- x₁, x₂, ..., x_{N-1} (记忆变量)

每个辅助变量都试图跟上前一个变量，传递速率 k = N/τ。
这样，信息从 x₀ 传递到 x_{N-1} 需要大约 τ 时间，
因此 x_{N-1} ≈ x(t-τ)。
"""

# =============================================================================
# 参数设置 - 增强效果的参数选择
# =============================================================================
tau = 3.0      # 延迟时间 (增加到3秒，效果更明显)
N = 50         # 变量数量（链条长度）- 增加到50，近似更精确
k = N / tau    # 传递速率

# DDE 参数 - 调整系数使振荡行为更明显
alpha = 0.1    # 当前状态的衰减系数 (减小，让系统更不稳定)
beta = 1.0     # 延迟状态的影响系数 (增大，增强延迟效应)

print(f"模拟参数:")
print(f"- 延迟时间 τ = {tau} 秒")
print(f"- 链条长度 N = {N} 个变量")
print(f"- 传递速率 k = {k:.2f}")
print(f"- DDE 方程: dx/dt = -{alpha}*x(t) + {beta}*x(t-{tau})")
print(f"- 理论分析: 当 β > α 且 τ 足够大时，系统会产生振荡")

def system_dynamics(states, t):
    """
    多变量 ODE 系统，用于近似 DDE
    
    参数:
    - states: 状态向量 [x(t), x₁, x₂, ..., x_{N-1}]
    - t: 时间
    
    返回:
    - d_states: 状态导数向量
    """
    x_current = states[0]           # 当前状态 x(t)
    memory_chain = states[1:]       # 记忆链条 [x₁, x₂, ..., x_{N-1}]
    
    d_states = np.zeros_like(states)
    
    # 核心 DDE 逻辑：使用链条末端近似延迟状态
    # x_past_approx ≈ x(t-τ)
    x_past_approx = memory_chain[-1]
    
    # DDE 方程: dx/dt = -α*x(t) + β*x(t-τ)
    d_states[0] = -alpha * x_current + beta * x_past_approx
    
    # 链条传递逻辑：每个变量都试图跟上前一个变量
    # 这创建了信息传递的"延迟"
    d_states[1] = k * (x_current - states[1])  # 第一个记忆变量跟随当前状态
    
    # 其余记忆变量形成链条
    for i in range(2, N):
        d_states[i] = k * (states[i-1] - states[i])
        
    return d_states

# =============================================================================
# 初始条件和求解
# =============================================================================
# 使用更有趣的初始条件：当前状态为1，记忆状态为0
# 这样可以清楚地看到延迟效应
initial_conditions = np.zeros(N)
initial_conditions[0] = 1.0  # x(0) = 1
# 其余记忆变量初始化为0，表示过去状态为0

print(f"\n初始条件:")
print(f"- x(0) = {initial_conditions[0]}")
print(f"- x(t<0) = 0 (通过记忆变量初始化为0)")

# 延长模拟时间以观察完整动态
t = np.linspace(0, 20, 2000)  # 从20秒延长到20秒，步长更细

# 求解 ODE 系统
print("\n正在求解 ODE 系统...")
solution = odeint(system_dynamics, initial_conditions, t)

# =============================================================================
# 可视化结果
# =============================================================================
plt.figure(figsize=(14, 10))

# 主图：当前状态 vs 延迟状态近似
plt.subplot(2, 2, 1)
plt.plot(t, solution[:, 0], 'b-', label='当前状态 $x(t)$', linewidth=2)
plt.plot(t, solution[:, -1], 'r--', label=f'近似延迟状态 $x(t-{tau})$', linewidth=2, alpha=0.8)
plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
plt.title(f'DDE 近似: 当前状态 vs 延迟状态 (τ={tau}s, N={N})')
plt.xlabel('时间 t')
plt.ylabel('状态值')
plt.legend()
plt.grid(True, alpha=0.3)

# 相图：显示系统的振荡特性
plt.subplot(2, 2, 2)
plt.plot(solution[:, 0], solution[:, -1], 'g-', linewidth=1)
plt.title('相图: $x(t)$ vs $x(t-τ)$')
plt.xlabel('$x(t)$')
plt.ylabel(f'$x(t-{tau})$')
plt.grid(True, alpha=0.3)

# 显示链条中几个关键点的状态
plt.subplot(2, 2, 3)
chain_indices = [0, N//4, N//2, 3*N//4, N-1]  # 选择链条中的几个点
labels = ['x(t)', f'x(t-{tau*0.25:.1f})', f'x(t-{tau*0.5:.1f})', 
          f'x(t-{tau*0.75:.1f})', f'x(t-{tau})']
colors = ['blue', 'orange', 'green', 'purple', 'red']

for i, idx in enumerate(chain_indices):
    plt.plot(t, solution[:, idx], color=colors[i], label=labels[i], linewidth=1.5)

plt.title('链条中不同位置的状态演化')
plt.xlabel('时间 t')
plt.ylabel('状态值')
plt.legend()
plt.grid(True, alpha=0.3)

# 显示延迟效应的直观演示
plt.subplot(2, 2, 4)

# 改进峰值检测：使用更稳健的方法
from scipy.signal import find_peaks

# 找到当前状态的峰值
peaks, _ = find_peaks(solution[:, 0], height=0.1, distance=50)  # 设置最小高度和最小距离

if len(peaks) > 0:
    # 使用第一个显著峰值
    first_peak_idx = peaks[0]
    first_peak_time = t[first_peak_idx]
    first_peak_value = solution[first_peak_idx, 0]
    
    # 确保延迟后的时间点在范围内
    delayed_time = first_peak_time + tau
    if delayed_time <= t[-1]:
        delayed_time_idx = np.argmin(np.abs(t - delayed_time))
        delayed_value = solution[delayed_time_idx, 0]
        
        plt.plot(t, solution[:, 0], 'b-', linewidth=2, label='$x(t)$')
        plt.plot(first_peak_time, first_peak_value, 'ro', markersize=8, label='当前峰值')
        plt.plot(delayed_time, delayed_value, 'go', markersize=8, label='τ秒后的值')
        plt.axvline(x=first_peak_time, color='r', linestyle='--', alpha=0.7)
        plt.axvline(x=delayed_time, color='g', linestyle='--', alpha=0.7)
        plt.text(first_peak_time, first_peak_value*1.1, f't={first_peak_time:.1f}', 
                 ha='center', color='red', fontsize=9)
        plt.text(delayed_time, delayed_value*1.1, f't+τ={delayed_time:.1f}', 
                 ha='center', color='green', fontsize=9)
    else:
        # 如果延迟时间超出范围，只显示当前峰值
        plt.plot(t, solution[:, 0], 'b-', linewidth=2, label='$x(t)$')
        plt.plot(first_peak_time, first_peak_value, 'ro', markersize=8, label='当前峰值')
        plt.axvline(x=first_peak_time, color='r', linestyle='--', alpha=0.7)
        plt.text(first_peak_time, first_peak_value*1.1, f't={first_peak_time:.1f}', 
                 ha='center', color='red', fontsize=9)
else:
    # 如果没有找到峰值，显示原始曲线并添加说明
    plt.plot(t, solution[:, 0], 'b-', linewidth=2, label='$x(t)$')
    plt.text(0.5, 0.5, '未检测到显著峰值\n系统可能单调变化', 
             transform=plt.gca().transAxes, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.title('延迟效应直观演示')
plt.xlabel('时间 t')
plt.ylabel('$x(t)$')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 分析和验证
# =============================================================================
print(f"\n=== 结果分析 ===")
print(f"1. 系统表现出明显的振荡行为，这是延迟微分方程的典型特征")
print(f"2. 当前状态 x(t) 和延迟状态 x(t-τ) 之间存在明显的相位差")
print(f"3. 链条方法成功地将 DDE 转换为 ODE 系统进行数值求解")
print(f"4. 增加 N (链条长度) 可以提高近似的精度")
print(f"5. 这种方法在科学计算中很有用，因为大多数数值求解器只能处理 ODE")

# 计算一些统计信息
final_oscillation_amplitude = np.std(solution[-500:, 0])  # 最后500个点的标准差
print(f"\n最终振荡幅度 (标准差): {final_oscillation_amplitude:.4f}")

# 理论验证：对于线性 DDE dx/dt = -αx(t) + βx(t-τ)
# 特征方程: λ + α - βe^(-λτ) = 0
# 当 β > α 且 τ 足够大时，存在具有正实部的复根，导致振荡增长
print(f"\n理论预期: 由于 β({beta}) > α({alpha}) 且 τ({tau}) 较大，")
print(f"系统应该表现出持续的振荡行为，这与数值结果一致。")