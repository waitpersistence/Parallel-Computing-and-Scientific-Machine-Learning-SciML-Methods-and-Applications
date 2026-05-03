using StaticArrays
using Plots
using ComponentArrays

# =============================================================================
# 参数定义和Butcher表格
# =============================================================================

# Lotka-Volterra模型参数
const BASE_PARAMS = (alpha = 1.5, beta = 1.0, gamma = 3.0, delta = 1.0)
const INITIAL_PARAMS = (alpha = 1.2, beta = 0.8, gamma = 2.8, delta = 0.8)
const DT = 0.25
const U0 = [1.0, 1.0]
const T_SPAN = (0.0, 10.0)

# Dormand-Prince 5(4)方法的Butcher表格
const C = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
const A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
]
const B = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

# =============================================================================
# 基础Lotka-Volterra方程
# =============================================================================

"""
Lotka-Volterra捕食者-猎物模型
- u: 状态向量 [猎物数量, 捕食者数量]
- p: 参数元组 (alpha, beta, gamma, delta)
- t: 时间（此模型为自治系统，不显式依赖时间）
"""
function lotka_volterra(u, p, t)
    x, y = u[1], u[2]
    dx = p.alpha * x - p.beta * x * y
    dy = -p.gamma * y + p.delta * x * y
    return [dx, dy]
end

# =============================================================================
# Dormand-Prince 5阶Runge-Kutta步进函数
# =============================================================================

"""
Dormand-Prince 5阶Runge-Kutta单步计算
"""
function dp5_step(f, u, p, t, dt)   
    k1 = f(u, p, t)
    k2 = f(u + k1 * A[2][1] * dt, p, t + c[2] * dt)
    k3 = f(u + k1 * A[3][1] * dt + k2 * A[3][2] * dt, p, t + c[3] * dt)
    k4 = f(u + k1 * A[4][1] * dt + k2 * A[4][2] * dt + k3 * A[4][3] * dt, p, t + c[4] * dt)
    k5 = f(u + k1 * A[5][1] * dt + k2 * A[5][2] * dt + k3 * A[5][3] * dt + k4 * A[5][4] * dt, p, t + c[5] * dt) 
    k6 = f(u + k1 * A[6][1] * dt + k2 * A[6][2] * dt + k3 * A[6][3] * dt + k4 * A[6][4] * dt + k5 * A[6][5] * dt, p, t + c[6] * dt)
    k7 = f(u + k1 * A[7][1] * dt + k2 * A[7][2] * dt + k3 * A[7][3] * dt + k4 * A[7][4] * dt + k5 * A[7][5] * dt + k6 * A[7][6] * dt, p, t + c[7] * dt)
    
    u_next = u + dt * (b[1] * k1 + b[2] * k2 + b[3] * k3 + b[4] * k4 + b[5] * k5 + b[6] * k6 + b[7] * k7)
    return u_next
end

# =============================================================================
# 扩展系统（包含参数敏感性）
# =============================================================================

"""
扩展的Lotka-Volterra系统，用于计算参数敏感性
返回值为ComponentArray，包含状态导数和敏感度导数
"""
function augmented_f_oop(w, p, t)
    x, y = w.u
    S = w.S
    alpha, beta, gamma, delta = p

    # 原系统导数
    dx = alpha * x - beta * x * y
    dy = -gamma * y + delta * x * y
    du = [dx, dy]

    # 雅可比矩阵
    Ju = [alpha - beta * y    -beta * x;
          delta * y           -gamma + delta * x]
    
    Jp = [x   -x * y   0    0;
          0    0      -y   x * y]

    # 敏感度演化方程: dS/dt = Ju * S + Jp
    dS = Ju * S + Jp

    return ComponentArray(u = du, S = dS)
end

# =============================================================================
# 数值求解器
# =============================================================================

"""
使用DP5方法求解普通微分方程
"""
function solve_ode(f, u0, p, tspan, dt)
    ts = tspan[1]:dt:tspan[2]
    results = Vector{typeof(u0)}(undef, length(ts))
    results[1] = u0
    
    for i in 1:(length(ts) - 1)
        t = ts[i]
        u_next = dp5_step(f, results[i], p, t, dt)
        results[i+1] = u_next
    end
    
    return ts, results
end

"""
使用DP5方法求解扩展系统（包含敏感性）
"""
function solve_augmented_system(p, tspan, dt)
    S0 = zeros(2, 4)
    w0 = ComponentArray(u = U0, S = S0)
    ts, results = solve_ode(augmented_f_oop, w0, p, tspan, dt)
    return ts, results
end

# =============================================================================
# 主程序执行
# =============================================================================

# 第一部分：基础Lotka-Volterra系统求解
println("=== 第一部分：基础Lotka-Volterra系统 ===")
ts_basic = T_SPAN[1]:DT:T_SPAN[2]
results_basic = zeros(2, length(ts_basic))
results_basic[:, 1] = U0

for (i, t) in enumerate(ts_basic[1:end-1])
    u_next = dp5_step(lotka_volterra, @view(results_basic[:, i]), BASE_PARAMS, t, DT)
    results_basic[:, i+1] = u_next
end

# 绘制基础系统结果
p1 = plot(ts_basic, results_basic[1, :], label="Prey (x)", xlabel="Time", ylabel="Population", lw=2)
plot!(p1, ts_basic, results_basic[2, :], label="Predator (y)", lw=2)
display(p1)

p2 = plot(results_basic[1, :], results_basic[2, :], 
          label="Phase Path", xlabel="Prey", ylabel="Predator", 
          aspect_ratio=:equal, title="Phase Portrait")
display(p2)

# 第二部分：参数敏感性分析
println("=== 第二部分：参数敏感性分析 ===")
ts_sens, results_sens = solve_augmented_system(BASE_PARAMS, T_SPAN, DT)

# 提取状态和敏感度数据
x_vals = [w.u[1] for w in results_sens]
y_vals = [w.u[2] for w in results_sens]
s11_vals = [w.S[1, 1] for w in results_sens]  # ∂x/∂α
s23_vals = [w.S[2, 3] for w in results_sens]  # ∂y/∂γ

# 绘制敏感性结果
p3 = plot(ts_sens, x_vals, label="Prey (x)", xlabel="Time", ylabel="Population", lw=2)
plot!(p3, ts_sens, y_vals, label="Predator (y)", lw=2)

p4 = plot(ts_sens, s11_vals, label="Sensitivity ∂x/∂α", xlabel="Time", ylabel="Sensitivity Value", lw=1.5)
plot!(p4, ts_sens, s23_vals, label="Sensitivity ∂y/∂γ", lw=1.5)

sensitivity_plot = plot(p3, p4, layout=(2, 1), size=(800, 600))
display(sensitivity_plot)

# 第三部分：参数优化（梯度下降）
println("=== 第三部分：参数优化 ===")
ts_opt, results_target = solve_augmented_system(BASE_PARAMS, T_SPAN, DT)  # 目标轨迹
newp = ComponentVector(INITIAL_PARAMS)
param_history = []

eta = 1e-5
n_iterations = 1000

for iter in 1:n_iterations
    # 生成当前参数下的轨迹
    _, results_current = solve_augmented_system(newp, T_SPAN, DT)
    
    # 计算损失和梯度
    loss = 0.0
    grad = zeros(4)
    for i in 1:length(ts_opt)
        res = results_current[i].u - results_target[i].u
        loss += 0.5 * sum(res .^ 2)
        grad += vec(res' * results_current[i].S)
    end
    
    if iter % 100 == 0
        println("Iteration $iter: Loss = $loss")
    end
    
    # 记录参数历史
    push!(param_history, copy(newp))
    
    # 梯度下降更新
    newp .= newp .- eta * grad
end

println("优化完成！")
println("初始参数: ", INITIAL_PARAMS)
println("最终参数: ", (alpha = newp.alpha, beta = newp.beta, gamma = newp.gamma, delta = newp.delta))
println("目标参数: ", BASE_PARAMS)

# 绘制参数优化过程
iterations = 1:length(param_history)
param_plot = plot(iterations, [param_history[i].alpha for i in 1:length(param_history)], 
                  label="alpha", xlabel="Iteration", ylabel="Parameter Value", lw=2)
plot!(param_plot, iterations, [param_history[i].beta for i in 1:length(param_history)], label="beta", lw=2)
plot!(param_plot, iterations, [param_history[i].gamma for i in 1:length(param_history)], label="gamma", lw=2)
plot!(param_plot, iterations, [param_history[i].delta for i in 1:length(param_history)], label="delta", lw=2)
display(param_plot)