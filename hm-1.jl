using Distributions
using Plots
using Distributed
using SharedArrays
# ==============================================================================
# Part 1: 牛顿法分位数计算
# ==============================================================================

function _quantile_newton(dist, y::T, x0) where T<:AbstractFloat
    x = T(x0)
    tol = eps(T) * 10
    max_iter = 100
    
    # 检查初始值是否合适，如果不合适则调整到合理的初始猜测
    pdf_val = pdf(dist, x)
    if abs(pdf_val) < eps(T)
        # 初始值不合适，尝试使用分布的均值作为初始猜测
        try
            x = mean(dist)
            pdf_val = pdf(dist, x)
            if abs(pdf_val) < eps(T)
                # 如果均值也不合适，尝试中位数
                x = median(dist)
            end
        catch
            # 如果无法计算均值或中位数，使用一个默认的合理值
            x = T(1.0)
        end
    end
    
    for i in 1:max_iter
        # Newton step
        # x=x-f(x)/f'(x)
        # 计算当前cdf及误差
        current_cdf = cdf(dist, x)
        pdf_val = pdf(dist, x)
        if abs(pdf_val) < eps(T)   
            @warn "PDF is too small, stopping iteration to avoid division by zero."
            break
        end
        
        # Newton 更新
        delta = (current_cdf - y) / pdf_val
        x -= delta
        if abs(delta) < tol
            return x  # 收敛成功，返回结果
        end
    end
    # 如果没收敛，抛出错误
    error("未能收敛。最后一次 delta: ", abs((cdf(dist, x) - y) / pdf(dist, x)))
end

function quantile_newton(dist, y, x0)
    T = promote_type(typeof(y),typeof(x0),Float64)
    return _quantile_newton(dist, T(y), T(x0)) 
end

# 测试牛顿法分位数计算
println("=== 牛顿法分位数计算测试 ===")
d = Normal(0, 1)
res = quantile_newton(d, 0.95, 0.0)
println("计算结果: ", res)
println("精度验证 (CDF(res)): ", cdf(d, res))

test_cases = [
    (0.95, Normal(0, 1), 0.0),
    (0.5, Gamma(5, 1), 0.0),
    (0.8, Beta(2, 4), 0.0)
]

for (y, d, x0) in test_cases
    mine = quantile_newton(d, y, x0)
    theirs = quantile(d, y)
    dist_name = string(typeof(d))
    println("分布: $dist_name, 目标概率: $y")
    println("  我的结果: $mine")
    println("  库的结果: $theirs")
    println("  误差: $(abs(mine - theirs))")
end

# ==============================================================================
# Part 2: 分岔图生成器 - 基础函数
# ==============================================================================

function calc_attractor!(out, f, p, num_attract=150, warmup=400)
    x = 0.25
    # 1. 预热阶段：只迭代，不记录结果
    for _ in 1:warmup
        x = f(x, p)
    end
    # 2. 采样阶段：迭代并存入 out 向量
    for i in 1:num_attract 
        x = f(x, p)
        out[i] = x
    end
    return out
end

# 定义逻辑斯谛映射函数
logistic(x, r) = r * x * (1 - x)

# 验证基础函数
out = zeros(150)
calc_attractor!(out, logistic, 2.9, 150, 400)
println("\n=== 基础 attractor 函数验证完成 ===")

# ==============================================================================
# Part 3: 多线程版本
# ==============================================================================

function bifurcation_threaded(r_range, n_samples=150, warmup=400)
    n_r = length(r_range)
    out_matrix = Matrix{Float64}(undef, n_samples, n_r)
    
    Threads.@threads for i in 1:n_r
        r = r_range[i]
        out = zeros(n_samples)
        calc_attractor!(out, logistic, r, n_samples, warmup)
        out_matrix[:, i] = out
    end
    
    return out_matrix
end

# ==============================================================================
# Part 4: 多进程版本实现
# ==============================================================================

# 独立的纯函数用于多进程计算 (在主进程定义，随后分发)
function calc_single_attractor(r, n_samples=150, warmup=400)
    out = zeros(n_samples)
    x = 0.25
    # 预热阶段
    for _ in 1:warmup
        x = logistic(x, r)
    end
    # 采样阶段
    for i in 1:n_samples
        x = logistic(x, r)
        out[i] = x
    end
    return out
end


    # 确保有 worker 进程
    if nworkers() == 0
        addprocs(min(Sys.CPU_THREADS - 1, 4))  # 限制最多4个额外进程
    end
    
   # 广播环境到所有 worker
@everywhere begin
    using Distributed
    
    function logistic_mp(x, r)
        return r * x * (1 - x)
    end
    
    function calc_single_attractor_mp(r, n_samples, warmup)
        out = zeros(n_samples)#局部变量out
        x = 0.25
        for _ in 1:warmup
            x = logistic_mp(x, r)
        end
        for i in 1:n_samples
            x = logistic_mp(x, r)
            out[i] = x
        end
        return out
    end
end
    
    println("开始 pmap 计算...")
# --- 第二步：定义执行函数 ---

function bifurcation_pmap(r_range, n_samples=150, warmup=400)
    println("开始 pmap 计算...")
    # pmap 会自动收集结果并返回一个 Vector of Vectors
    results = pmap(r -> calc_single_attractor_mp(r, n_samples, warmup), r_range)
# pmap 的工作原理：

# 自动并行化：pmap 会将 r_range 中的每个 r 值分配给可用的 worker 进程
# 匿名函数：r -> calc_single_attractor_mp(r, n_samples, warmup) 是一个 lambda 函数，对每个 r 调用计算函数
# 结果收集：pmap 自动收集所有 worker 进程返回的结果，形成一个向量的向量（Vector of Vectors）

    return hcat(results...)
end

# ==========================================
# 2. 逻辑执行区 (这里只负责发号施令)
# ==========================================

function bifurcation_distributed(r_range, n_samples=150, warmup=400)
    n_r = length(r_range)
    
    # 只要在顶层 using 了 SharedArrays，这里就能直接用 SharedMatrix
    out_matrix = SharedMatrix{Float64}(n_samples, n_r)
    
    println("开始 @distributed 计算...")
    
    # @sync 保证主进程会等所有 Worker 算完
    @sync @distributed for i in 1:n_r
        r = r_range[i]
        result = calc_single_attractor_mp(r, n_samples, warmup)
        # SharedMatrix 的写入是原位的，所有进程都在填这张大表
        out_matrix[:, i] = result
    end
    
    # 用 sdata 把共享内存转回普通矩阵，方便后续绘图
    return sdata(out_matrix)
end

# ==============================================================================
# Part 5: 性能比较和可视化
# ==============================================================================

println("\n=== 分岔图生成性能比较 ===")

# 参数设置（使用较小的范围进行快速测试）
r_range_test = 2.9:0.01:4.0  # 步长 0.01，快速测试
n_samples_test = 100

# 多线程版本
println("1. 测试多线程版本...")
thread_start = time()
thread_result = bifurcation_threaded(r_range_test, n_samples_test)
thread_time = time() - thread_start
println("   多线程耗时: $(round(thread_time, digits=3)) 秒")

# 多进程 pmap 版本
println("2. 测试 pmap 多进程版本...")
pmap_start = time()
pmap_result = bifurcation_pmap(r_range_test, n_samples_test)
pmap_time = time() - pmap_start
println("   pmap 耗时: $(round(pmap_time, digits=3)) 秒")

# 多进程 @distributed 版本
println("3. 测试 @distributed 多进程版本...")
dist_start = time()
dist_result = bifurcation_distributed(r_range_test, n_samples_test)
dist_time = time() - dist_start
println("   @distributed 耗时: $(round(dist_time, digits=3)) 秒")

# 结果验证
println("\n=== 结果一致性验证 ===")
println("多线程 vs pmap: ", isapprox(thread_result, pmap_result, atol=1e-10))
println("多线程 vs @distributed: ", isapprox(thread_result, dist_result, atol=1e-10))

# 性能总结
println("\n=== 性能总结 ===")
times = [thread_time, pmap_time, dist_time]
labels = ["多线程", "pmap", "@distributed"]
best_idx = argmin(times)
println("最快的方法: $(labels[best_idx]) ($(round(times[best_idx], digits=3)) 秒)")

# 生成完整分岔图（可选，注释掉以节省时间）
# println("\n=== 生成完整分岔图 ===")
# full_r_range = 2.9:0.001:4.0
# final_result = bifurcation_threaded(full_r_range, 150)

# r_plot = [r for r in full_r_range for _ in 1:150]
# x_plot = vec(final_result)

# scatter(r_plot, x_plot, 
#     markersize=0.1, 
#     markeralpha=0.2, 
#     legend=false, 
#     color=:black,
#     xlabel="r", ylabel="Steady State x",
#     title="Bifurcation Diagram of the Logistic Map")

# 如果你用 @distributed，你需要预先开辟一个 SharedMatrix，让每个进程往自己对应的列里写数据。

# 如果你用 pmap，你可以直接写成 results = pmap(r -> calc_attractor(r), r_range)，然后把返回的列表拼成矩