function forward(u,W1,b1,W2,b2)
    z1 = W1*u+b1
    a1=tanh.(z1)
    out=W2*a1+b2
    return out
end

function backward(u, W1, b1, W2, b2, y, a1, z1)
    # 1. 最后一层 (Linear 2)
    g_b2 = y
    g_W2 = y * a1'       # 输出对 W2 的梯度
    g_a1 = W2' * y       # 输出对 a1 的梯度

    # 2. 激活层 (tanh) - 链式法则的核心步骤
    g_z1 = g_a1 .* (1 .- a1.^2) 

    # 3. 第一层 (Linear 1) - 使用 g_z1 继续向上传递
    g_b1 = g_z1          # 输出对 b1 的梯度
    g_W1 = g_z1 * u'     # 输出对 W1 的梯度
    g_u  = W1' * g_z1    # 输出对输入 u 的梯度 (也就是 B_NN^u)

    return g_u, g_W1, g_b1, g_W2, g_b2
end


function nn_with_pullback(u,W1,b1,W2,b2)
    # 1. 前向传播：记录所有中间变量
    z1 = W1 * u + b1
    a1 = tanh.(z1)
    out = W2 * a1 + b2

    # 2. 返回输出值和 Pullback 闭包
    function pullback(y)
            # 1. 最后一层 (Linear 2)
        g_b2 = y
        g_W2 = y * a1'       # 输出对 W2 的梯度
        g_a1 = W2' * y       # 输出对 a1 的梯度

        # 2. 激活层 (tanh) - 链式法则的核心步骤
        g_z1 = g_a1 .* (1 .- a1.^2) 

        # 3. 第一层 (Linear 1) - 使用 g_z1 继续向上传递
        g_b1 = g_z1          # 输出对 b1 的梯度
        g_W1 = g_z1 * u'     # 输出对 W1 的梯度
        g_u  = W1' * g_z1    # 输出对输入 u 的梯度 (也就是 B_NN^u)

    
        return (g_u, g_W1, g_b1, g_W2, g_b2)
    end

    return out, pullback
end

using ForwardDiff
using LinearAlgebra

# 初始化参数
u  = rand(2)
W1 = rand(50, 2)
b1 = rand(50)
W2 = rand(2, 50)
b2 = rand(2)
# 定义标量化函数以便 ForwardDiff 处理
target_f(u_in) = sum(forward(u_in, W1, b1, W2, b2))

# 使用 ForwardDiff 计算参考梯度
grad_u_ref = ForwardDiff.gradient(target_f, u)
# 运行你完整封装的函数
out, pb = nn_with_pullback(u, W1, b1, W2, b2)

# 传入 y = [1.0, 1.0]，因为 sum 函数对每个分量的导数都是 1
grads = pb([1.0, 1.0]) 

# 提取对 u 的梯度（假设你在元组第一个位置返回了 g_u）
grad_u_manual = grads[1]

# 计算绝对误差
error_val = norm(grad_u_manual - grad_u_ref)
@show grad_u_manual,grad_u_ref
println("手动梯度与自动微分的误差为: ", error_val)

if error_val < 1e-12
    println("✅ 验证通过！你的 Pullback 实现非常精准。")
else
    println("❌ 验证失败。请检查矩阵转置或 tanh 导数环节。")
end
@show out




using OrdinaryDiffEq, LinearAlgebra, Statistics, ForwardDiff

# ==========================================
# 1. 数据生成 (Part 4 目标系统)
# ==========================================
u0 = [2.0, 0.0]
tspan = (0.0, 1.0)
t_data = 0.0:0.1:1.0
A = [-0.1  2.0; 
     -2.0 -0.1]

# 真实物理系统的 ODE
f_true(u, p, t) = A * u
prob_true = ODEProblem(f_true, u0, tspan)
sol_true = solve(prob_true, Tsit5(), saveat=t_data)
u_data = sol_true.u  # 获取真实轨迹作为训练目标

# ==========================================
# 2. 神经网络与参数管理 (Part 2)
# ==========================================
# 维度：u(2), W1(50x2), b1(50), W2(2x50), b2(2)
function unpack_p(p)
    W1 = reshape(p[1:100], 50, 2)
    b1 = p[101:150]
    W2 = reshape(p[151:250], 2, 50)
    b2 = p[251:252]
    return W1, b1, W2, b2
end

# 你在 Part 2 实现的带有 Pullback 的前向传播
function nn_with_pullback(u, W1, b1, W2, b2)
    z1 = W1 * u + b1
    a1 = tanh.(z1)
    out = W2 * a1 + b2
    
    # 定义 Pullback 闭包
    function pb(y)
        g_b2 = y
        g_W2 = y * a1'
        g_a1 = W2' * y
        g_z1 = g_a1 .* (1 .- a1.^2) # tanh 的导数
        g_b1 = g_z1
        g_W1 = g_z1 * u'
        g_u  = W1' * g_z1
        return (g_u, g_W1, g_b1, g_W2, g_b2)
    end
    return out, pb
end

# ==========================================
# 3. 伴随方法实现 (Part 3)
# ==========================================
function compute_gradient_adjoint(u0, p, t_data, u_target)
    W1, b1, W2, b2 = unpack_p(p)
    
    # --- 前向过程 ---
    # 注意：不使用 saveat 以保证高阶插值精度
    # 修正后（使用标准 function 格式，更易读且无歧义）：
    function f_node(u, p, t)
        W1, b1, W2, b2 = unpack_p(p)
        return W2 * tanh.(W1 * u + b1) + b2
    end
    prob_fwd = ODEProblem(f_node, u0, (t_data[1], t_data[end]), p)
    sol_fwd = solve(prob_fwd, Tsit5(), dense=true)
    
    # --- 初始化伴随变量 ---
    # 我们从 T 回溯到 0。由于有多个数据点，我们需要处理 λ 的 "Jump"
    λ = zeros(length(u0))
    μ = zeros(length(p))
    
    # 逆向时间循环（处理多个数据点）
    for i in length(t_data):-1:2
        t_start = t_data[i]
        t_end = t_data[i-1]
        
        # 1. 在数据点处增加梯度跳跃 (L2 Loss: ∂C/∂u = u_node - u_target)
        u_at_t = sol_fwd(t_start)
        λ .+= (u_at_t - u_target[i]) 
        
        # 2. 定义伴随 ODE 的右端项
        function adj_dynamics!(dstate, state, p, t)
        curr_λ = state[1:2]
        u_t = sol_fwd(t) 
        
        # 确保解包出的变量名与下方 pb 调用一致
        W1_curr, b1_curr, W2_curr, b2_curr = unpack_p(p)
        
        # 获取 pullback 闭包
        _, pb = nn_with_pullback(u_t, W1_curr, b1_curr, W2_curr, b2_curr)
        
        # 执行拉回
        g_u, g_W1, g_b1, g_W2, g_b2 = pb(curr_λ)
        
        # 原位更新 dstate
        dstate[1:2] .= -g_u
        dstate[3:end] .= -vcat(vec(g_W1), vec(g_b1), vec(g_W2), vec(g_b2))
        end
        
        # 3. 求解一段伴随方程
        curr_state = vcat(λ, μ)
        adj_prob = ODEProblem(adj_dynamics!, curr_state, (t_start, t_end), p)
        adj_sol = solve(adj_prob, Tsit5())
        
        # 更新 λ 和 μ 进入下一个区间
        λ .= adj_sol.u[end][1:2]
        μ .= adj_sol.u[end][3:end]
    end
    
    # 最后处理 t=0 时刻的跳跃
    λ .+= (sol_fwd(t_data[1]) - u_target[1])
    
    return μ, sol_fwd # 返回总梯度和前向解
end

# ==========================================
# 4. 训练循环 (Part 4)
# ==========================================
p_train = randn(252) * 0.1 # 随机初始化所有参数
lr = 0.01                  # 学习率
epochs = 100

println("开始训练 Neural ODE...")
for epoch in 1:epochs
    grad, sol_fwd = compute_gradient_adjoint(u0, p_train, t_data, u_data)
    
    # 简单的梯度下降
    p_train .-= lr .* grad
    
    # 计算当前 Loss (L2 距离)
    current_u = [sol_fwd(t) for t in t_data]
    loss = sum(norm.(current_u .- u_data))
    
    if epoch % 10 == 0
        println("Epoch $epoch, Loss: $loss")
    end
end