# =============================================================================
# Lesson 2: 科学机器学习与物理信息神经网络
# =============================================================================

using Lux, Optimisers, Enzyme, ComponentArrays, Random, Statistics, Plots

# =============================================================================
# 1. 神经网络基础：手动实现 vs Lux 框架
# =============================================================================
println("=== 1. 神经网络基础 ===")

# 手动实现简单神经网络
W = [randn(32,10), randn(32,32), randn(5,32)]
b = [zeros(32), zeros(32), zeros(5)]
simpleNN(x) = W[3]*tanh.(W[2]*tanh.(W[1]*x + b[1]) + b[2]) + b[3]
println("手动实现 NN 输出:", simpleNN(rand(10)))

# 使用 Lux 框架 - 推荐方式
NN2 = Chain(Dense(10 => 32, tanh),
           Dense(32 => 32, tanh),
           Dense(32 => 5))
ps, st = Lux.setup(Xoshiro(0), NN2)
println("Lux 框架 NN 输出:", NN2(rand(Float32, 10), ps, st)[1])

# 不同激活函数示例
NN3 = Chain(Dense(10 => 32, x->x^2),
            Dense(32 => 32, x->max(0,x)),
            Dense(32 => 5))
ps, st = Lux.setup(Xoshiro(0), NN3)
println("自定义激活函数 NN 输出:", NN3(rand(Float32, 10), ps, st)[1])

# =============================================================================
# 2. 神经网络训练基础
# =============================================================================
println("\n=== 2. 神经网络训练基础 ===")

NN = Chain(Dense(10 => 32, tanh),
           Dense(32 => 32, tanh),
           Dense(32 => 5))
ps, st = Lux.setup(Xoshiro(0), NN)

# 定义损失函数
loss(p) = sum(abs2, sum(abs2, NN(rand(Float32, 10), p, st)[1] .- 1f0) for i in 1:100)
println("初始损失:", loss(ps))

# 创建训练状态
tstate = Lux.Training.TrainState(NN, ps, st, Adam(0.1f0))

# 定义优化损失函数
function opt_loss(model, p, st, x)
   pred, st_new = model(x, p, st)
   return sum(abs2, pred .- 1f0), st_new, ()
end

# 训练循环
println("开始训练...")
for epoch in 1:1000
    global tstate
    x = rand(Float32, 10, 128)
    _, _, _, tstate = Lux.Training.single_train_step!(
        AutoEnzyme(), opt_loss, x, tstate
    )
end
println("训练后损失:", loss(tstate.parameters))

# =============================================================================
# 3. 物理信息神经网络 (PINN) - 微分方程求解
# =============================================================================
println("\n=== 3. 物理信息神经网络 (PINN) ===")

# 辅助函数：标量转向量
scalar_to_vector(x::Number) = reshape([x], 1, 1)
scalar_to_vector(x::AbstractVector) = reshape(x, 1, length(x))
scalar_to_vector(x::AbstractMatrix) = x

# 创建用于ODE求解的神经网络
NNODE = Chain(WrappedFunction(scalar_to_vector),
              Dense(1 => 32, tanh),
              Dense(32 => 1))
ps, st = Lux.setup(Xoshiro(0), NNODE)
println("NNODE 输出:", NNODE(1.0f0, ps, st)[1])

# 自定义模型包装器
struct ModelWrapper{M} <: AbstractLuxWrapperLayer{:model}
    model::M
end

function (m::ModelWrapper)(t, p, st)
    y, st = m.model(t, p, st)
    return scalar_to_vector(t) .* y .+ 1f0, st
end

model = ModelWrapper(NNODE)
ps, st = Lux.setup(Xoshiro(0), model)
println("包装模型输出:", model(1.0f0, ps, st)[1])

# PINN 损失函数：匹配导数
function loss(p)
    ϵ = sqrt(eps(Float32))
    t = 0f0:1f-2:1f0
    t_plus_ϵ = t .+ ϵ
    diff = (model(t_plus_ϵ, p, st)[1] .- model(t, p, st)[1]) / ϵ
    return mean(abs2, diff .- cos.(Float32(2π) .* scalar_to_vector(t)))
end

# 训练 PINN
tstate = Lux.Training.TrainState(model, ps, st, Descent(0.01f0))

function opt_loss(model, p, st, ts)
    ϵ = sqrt(eps(Float32))
    ts_plus_ϵ = ts .+ ϵ

    y, st = model(ts, p, st)
    y_plus_ϵ, st = model(ts_plus_ϵ, p, st)

    diff = (y_plus_ϵ .- y) ./ ϵ
    return MSELoss()(diff, cos.(Float32(2π) .* ts)), st, (;)
end

println("训练 PINN...")
for epoch in 1:1000
    global tstate
    _, loss_val, _, tstate = Lux.Training.single_train_step!(
      AutoEnzyme(), opt_loss, reshape(collect(0f0:1f-2:1f0), 1, :), tstate
    )
    if epoch % 200 == 0 || epoch == 1
        @info "PINN Training" epoch loss_val
    end
end

# 可视化结果
t = 0f0:0.001f0:1f0
g(t, p) = model(t, p, tstate.states)[1]
plot(t, vec(g(t, tstate.parameters)), label="NN", linewidth=2)
plot!(t, 1.0 .+ sin.(2π.*t)/2π, label="True Solution", linewidth=2, linestyle=:dash)
title!("PINN: Solving ODE with Neural Networks")
xlabel!("t")
ylabel!("u(t)")

# =============================================================================
# 4. 从数据学习微分方程 - 力学系统识别
# =============================================================================
println("\n=== 4. 力学系统识别 ===")

using DifferentialEquations

# 定义真实力学系统
k = 1.0
force(dx, x, k, t) = -k*x + 0.1sin(x)
prob = SecondOrderODEProblem(force, 1.0, 0.0, (0.0, 10.0), k)
sol = solve(prob)

# 生成训练数据
t_data = 0:3.3:10
dataset = sol(t_data)
position_data = [state[2] for state in sol(t_data)]
force_data = [force(state[1], state[2], k, t) for state in sol(t_data)]

# 可视化真实力和测量数据
plot_t = 0:0.01:10
data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1], state[2], k, t) for state in data_plot]

plot(plot_t, force_plot, xlabel="t", label="True Force", linewidth=2)
scatter!(t_data, force_data, label="Force Measurements", markersize=4)
title!("Force Data Generation")

# =============================================================================
# 5. 纯数据驱动的力学习
# =============================================================================
println("\n=== 5. 纯数据驱动学习 ===")

NNForce = Chain(WrappedFunction(scalar_to_vector),
           Dense(1 => 32, tanh),
           Dense(32 => 1))
ps2, st2 = Lux.setup(Xoshiro(0), NNForce)

# 数据驱动损失函数
function loss_force(p)
    pos = Float32.(position_data)
    force = scalar_to_vector(Float32.(force_data))
    return MSELoss()(NNForce(pos, p, st2)[1], force)
end

println("初始数据驱动损失:", loss_force(ps2))

# 训练纯数据驱动模型
tstate2 = Lux.Training.TrainState(NNForce, ps2, st2, Descent(0.01f0))

function opt_loss_force(model, p, s, (position_data, force_data))
    pos = Float32.(position_data)
    force = scalar_to_vector(Float32.(force_data))
    force_pred, st = model(pos, p, s)
    return MSELoss()(force_pred, force), st, ()
end

println("训练纯数据驱动模型...")
for epoch in 1:1000
    global tstate2
    _, loss_val, _, tstate2 = Lux.Training.single_train_step!(
        AutoEnzyme(), opt_loss_force, (position_data, force_data), tstate2
    )
    if epoch % 200 == 0
        @info "Data-driven Training" epoch loss_val
    end
end

# =============================================================================
# 6. 物理约束学习 - 结合数据和物理先验
# =============================================================================
println("\n=== 6. 物理约束学习 ===")

# 简化物理模型（已知的物理规律）
force2(dx, x, k, t) = -k*x
prob_simplified = SecondOrderODEProblem(force2, 1.0, 0.0, (0.0, 10.0), k)
sol_simplified = solve(prob_simplified)

# 可视化简化模型 vs 真实模型
plot(sol, label=["Velocity" "Position"], linewidth=2)
plot!(sol_simplified, label=["Velocity Simplified" "Position Simplified"], 
      linestyle=:dash, linewidth=2)
title!("True vs Simplified Physics")

# 物理约束损失函数
random_positions = Float32[2rand()-1 for i in 1:100] # random values in [-1,1]

function loss_ode(p)
    positions = scalar_to_vector(Float32.(random_positions))
    return MSELoss()(NNForce(positions, p, tstate2.states)[1], Float32.(-k*positions))
end

λ = 0.1f0  # 物理约束权重
composed_loss(p) = loss_force(p) + λ*loss_ode(p)
println("组合损失初始值:", composed_loss(tstate2.parameters))

# 重置训练状态进行组合训练
tstate2 = Lux.Training.TrainState(NNForce, Lux.setup(Xoshiro(0), NNForce)..., Descent(0.01f0))

function opt_composed_loss(model, p, s, (pos, force, rand_pos))
    loss_force_val, s, _ = opt_loss_force(model, p, s, (pos, force))
    positions = scalar_to_vector(Float32.(rand_pos))
    force_pred, s = model(positions, p, s)
    loss_ode_val = MSELoss()(force_pred, Float32.(-k * positions))
    return (
        loss_force_val + λ * loss_ode_val,
        s, 
        (; loss_force=loss_force_val, loss_ode=loss_ode_val)
    )
end

println("训练物理约束模型...")
for epoch in 1:1000
    global tstate2
    rand_pos = 2 .* rand(Float32, 100) .- 1
    _, loss_val, stats, tstate2 = Lux.Training.single_train_step!(
        AutoEnzyme(), opt_composed_loss, (position_data, force_data, rand_pos), tstate2
    )
    if epoch % 200 == 0
        @info "Physics-informed Training" epoch loss_val stats.loss_force stats.loss_ode
    end
end

# =============================================================================
# 7. 结果可视化与比较
# =============================================================================
println("\n=== 7. 结果比较 ===")

learned_force_plot = vec(
    NNForce(Float32.(positions_plot), tstate2.parameters, tstate2.states)[1]
)

plot(plot_t, force_plot, xlabel="t", label="True Force", linewidth=2)
plot!(plot_t, learned_force_plot, label="Predicted Force", linewidth=2, linestyle=:dash)
scatter!(t_data, force_data, label="Force Measurements", markersize=4)
title!("Physics-informed Learning Results")
xlabel!("Time")
ylabel!("Force")

# =============================================================================
# 关键概念总结
# =============================================================================
println("""
=== 关键概念总结 ===

1. **Lux 框架优势**：
   - 自动参数管理
   - 类型稳定性保证
   - 与自动微分无缝集成

2. **物理信息神经网络 (PINN)**：
   - 将物理定律作为损失函数约束
   - 可以解决没有数据的区域
   - 提高泛化能力

3. **物理约束学习**：
   - 结合稀疏数据 + 物理先验知识
   - 权重参数 λ 控制数据vs物理的重要性
   - 在数据不足时特别有效

4. **科学机器学习核心思想**：
   - 不是替代物理模型，而是增强物理模型
   - 利用数据学习未知的物理项
   - 保持已知物理规律的结构
""")