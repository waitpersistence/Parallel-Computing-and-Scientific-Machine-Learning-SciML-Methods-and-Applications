# =============================================================================
# Lesson 16: 微分方程求解与统计可视化
# =============================================================================

# 导入必要的包
using OrdinaryDiffEq, Plots
using StatsPlots
using Distributions
using DiffEqBase.EnsembleAnalysis
using KernelDensity
using ParameterizedFunctions

# =============================================================================
# 1. Lotka-Volterra 捕食者-猎物模型（确定性参数）
# =============================================================================
println("=== 1. Lotka-Volterra 模型（确定性参数）===")

# 定义Lotka-Volterra微分方程系统
function lotka_volterra(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]  # 猎物增长 - 被捕食
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]  # 捕食者死亡 + 捕食增长
end

# 设置参数和初始条件
θ = [1.5, 1.0, 3.0, 1.0]  # [猎物增长率, 捕食率, 捕食者死亡率, 转化效率]
u0 = [1.0; 1.0]           # 初始种群数量 [猎物, 捕食者]
tspan = (0.0, 10.0)       # 时间范围

# 创建并求解ODE问题
prob1 = ODEProblem(lotka_volterra, u0, tspan, θ)
sol = solve(prob1, Tsit5())
plot(sol, title="Lotka-Volterra 模型（确定性参数）")

# =============================================================================
# 2. Lotka-Volterra 模型（随机参数）
# =============================================================================
println("=== 2. Lotka-Volterra 模型（随机参数）===")

# 定义参数的先验分布
θ = [
    Uniform(0.5, 1.5),    # 猎物增长率的不确定性
    Beta(5, 1),           # 捕食率的不确定性  
    Normal(3, 0.5),       # 捕食者死亡率的不确定性
    Gamma(5, 2)           # 转化效率的不确定性
]

# 从先验分布中采样参数
_θ = rand.(θ)
prob1 = ODEProblem(lotka_volterra, u0, tspan, _θ)
sol = solve(prob1, Tsit5())
plot(sol, title="Lotka-Volterra 模型（随机参数）")

# =============================================================================
# 3. 集合模拟（Ensemble Simulation）
# =============================================================================
println("=== 3. 集合模拟 ===")

# 定义问题生成函数：每次生成新的随机参数
prob_func = function (prob, i, repeat)
    remake(prob, p = rand.(θ))
end

# 创建集合问题并求解
ensemble_prob = EnsembleProblem(ODEProblem(lotka_volterra, u0, tspan, θ),
                                prob_func = prob_func)
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 1000)

# 绘制集合摘要统计
plot(EnsembleSummary(sol), title="集合模拟结果摘要")

# =============================================================================
# 4. 统计分布可视化
# =============================================================================
println("=== 4. 统计分布可视化 ===")

# 创建正态分布
X = Normal(5, 1)

# 散点图和直方图
x = [rand(X) for i in 1:100]
scatter(x, [1 for i in 1:100], title="随机样本散点图")
histogram(x, title="样本直方图")

# 大样本直方图与理论分布对比
histogram([rand(X) for i in 1:10000], normed = true, 
          title="大样本直方图 vs 理论分布", label="样本")
plot!(X, lw = 5, label="理论分布")

# 核密度估计与理论分布对比
plot(kde([rand(X) for i in 1:10000]), lw = 5, 
     title="核密度估计 vs 理论分布", label="KDE")
plot!(X, lw = 5, label="理论分布")

# =============================================================================
# 5. 谐振子模型（使用ParameterizedFunctions）
# =============================================================================
println("=== 5. 谐振子模型 ===")

# 使用@ode_def宏定义谐振子方程
u0 = [1.0, 0.0]  # 初始位置和速度
harmonic! = @ode_def HarmonicOscillator begin
    dv = -x    # 加速度 = -位置（简谐运动）
    dx = v     # 速度 = 位置的导数
end

# 设置长时间模拟
tspan = (0.0, 10000.0)
prob = ODEProblem(harmonic!, u0, tspan)
sol = solve(prob, Tsit5())

# 设置绘图格式为PNG（处理大量数据点）
gr(fmt = :png)

# 相图（位置vs速度）和时间序列图
plot(sol, idxs = (1, 2), title="谐振子相图")
plot(sol, title="谐振子时间序列")