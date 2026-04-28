using DifferentialEquations
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using ForwardDiff, Plots

# 1. 定义物理模型 (ODE)
# u[1]: 位移 x, u[2]: 速度 v
# p[1]: 阻尼系数 c, p[2]: 劲度系数 k
function spring_damper!(du, u, p, t)
    x, v = u
    c, k = p
    m = 1.0  # 假设质量为 1kg
    du[1] = v
    du[2] = - (c/m)*v - (k/m)*x
end

# 2. 生成“实验数据” (Ground Truth)
u0 = [1.0, 0.0]        # 初始状态：位移1.0, 速度0
tspan = (0.0, 10.0)
true_p = [0.5, 2.0]    # 真正的物理参数：c=0.5, k=2.0
prob = ODEProblem(spring_damper!, u0, tspan, true_p)

# 模拟实验观测，并加上一点高斯噪声
tsteps = 0.0:0.1:10.0
sol_true = solve(prob, Tsit5(), saveat=tsteps)
noisy_data = sol_true[1, :] .+ 0.05 * randn(length(tsteps))

# 3. 定义损失函数 (Loss Function) —— 也就是“打靶”的误差
function loss(p, _)
    # “射击”：用当前的参数猜想 p 跑一遍模拟
    # 这里使用 remake 提高效率，避免重新创建对象
    _prob = remake(prob, p=p)
    sol = solve(_prob, Tsit5(), saveat=tsteps)
    
    # 如果模拟失败（参数太离谱导致发散），返回无穷大
    if sol.retcode != ReturnCode.Success
        return Inf
    end
    
    # 计算 L2 损失 (拟合位移 u[1])
    return sum(abs2, sol[1, :] .- noisy_data)
end

# 4. 配置优化问题
p_guess = [1.5, 0.5] # 随意的初始猜想
# 关键点：使用 AutoForwardDiff() 开启前向自动微分，这就是在算 Jacobian
adtype = Optimization.AutoForwardDiff()
optf = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optf, p_guess)

# 5. 执行优化 (寻优过程)
# 这里先用 Adam 快速靠近，再用 BFGS (二阶近似) 精准狙击
println("开始优化...")
res1 = solve(optprob, Adam(0.05), maxiters=100)
optprob2 = remake(optprob, u0=res1.u)
res2 = solve(optprob2, BFGS())

# 6. 结果展示
println("真实参数: ", true_p)
println("辨识参数: ", res2.u)

# 可视化比较
plt = plot(tsteps, noisy_data, seriestype=:scatter, label="Experimental Data (Noisy)")
sol_final = solve(remake(prob, p=res2.u), Tsit5(), saveat=tsteps)
plot!(plt, sol_final, vars=(0, 1), lw=3, label="Fitted Curve (Shooting Method)")
display(plt)