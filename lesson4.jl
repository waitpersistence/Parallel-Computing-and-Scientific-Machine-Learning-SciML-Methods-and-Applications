using Distributed
using StaticArrays, BenchmarkTools
using Base.Threads
using Statistics

# 启动 4 个工作进程（模拟 4 个灶台）
if nworkers() == 1
    addprocs(4) 
end

println("当前工人数量: ", nworkers())

# --- 对比 1：同步执行 (串行，一个接一个) ---
println("\n>>> 正在运行【同步版本】(老实排队)...")
@time begin
    results_sync = Vector{Any}(undef, nworkers())
    for (idx, pid) in enumerate(workers())
        # 没有 @async，主线程会在这里死等 2 秒，拿回结果才进下一次循环
        results_sync[idx] = remotecall_fetch(sleep, pid, 2)
    end
end
# 预期耗时：约 2秒 * 工人数 (比如 4个工人就是 8秒)

# --- 对比 2：异步执行 (并发，一起动手) ---
println("\n>>> 正在运行【异步版本】(火力全开)...")
@time begin
    results_async = Vector{Any}(undef, nworkers())
    @sync for (idx, pid) in enumerate(workers())
        # 有了 @async，主线程只管下令，发完 4 个指令就去休息
        @async results_async[idx] = remotecall_fetch(sleep, pid, 2)
    end
end
# 预期耗时：约 2秒 (不管你有几个工人，只要工人够多，都是 2秒)

# ==================== Lorenz 系统实现 ====================

function lorenz(u,p)
  α,σ,ρ,β = p
  @inbounds begin
    du1 = u[1] + α*(σ*(u[2]-u[1]))
    du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du3 = u[3] + α*(u[1]*u[2] - β*u[3])
  end
  @SVector [du1,du2,du3]
end

function solve_system_save!(u,f,u0,p,n)
  @inbounds u[1] = u0
  @inbounds for i in 1:length(u)-1
    u[i+1] = f(u[i],p)
  end
  u
end

function lorenz!(du,u,p)
  α,σ,ρ,β = p
  @inbounds begin
    du[1] = u[1] + α*(σ*(u[2]-u[1]))
    du[2] = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du[3] = u[3] + α*(u[1]*u[2] - β*u[3])
  end
end

function solve_system_save_iip!(u,f,u0,p,n)
  @inbounds u[1] = u0
  @inbounds for i in 1:length(u)-1
    f(u[i+1],u[i],p)
  end
  u
end

function lorenz_mt!(du,u,p)
  α,σ,ρ,β = p
  let du=du, u=u, p=p
    Threads.@threads for i in 1:3
      @inbounds begin
        if i == 1
          du[1] = u[1] + α*(σ*(u[2]-u[1]))
        elseif i == 2
          du[2] = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
        else
          du[3] = u[3] + α*(u[1]*u[2] - β*u[3])
        end
        nothing
      end
    end
  end
  nothing
end

# ==================== 性能测试 ====================

p = (0.02,10.0,28.0,8/3)

# 测试非原地版本
u1 = Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef,1000)
@btime solve_system_save!(u1,lorenz,@SVector([1.0,0.0,0.0]),p,1000)

# 测试原地版本
u2 = [Vector{Float64}(undef,3) for i in 1:1000]
@btime solve_system_save_iip!(u2,lorenz!,[1.0,0.0,0.0],p,1000)

# 测试多线程原地版本
u3 = [Vector{Float64}(undef,3) for i in 1:1000]
@btime solve_system_save_iip!(u3,lorenz_mt!,[1.0,0.0,0.0],p,1000);

# ==================== 轨迹均值计算 ====================

function compute_trajectory_mean(u0,p)
  u = Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef,1000)
  solve_system_save!(u,lorenz,u0,p,1000);
  mean(u)
end

@btime compute_trajectory_mean(@SVector([1.0,0.0,0.0]),p)

# 使用预分配数组的版本
u4 = Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef,1000)
function compute_trajectory_mean2(u0,p)
  # u is automatically captured
  solve_system_save!(u4,lorenz,u0,p,1000);
  mean(u4)
end

@btime compute_trajectory_mean2(@SVector([1.0,0.0,0.0]),p)

# ==================== 并行映射示例 ====================

function tmap(f,ps)
  out = Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef,length(ps))
  Threads.@threads for i in 1:length(ps)
    # each loop part is using a different part of the data
    out[i] = f(ps[i])
  end
  out
end

# 创建参数数组用于测试
test_ps = fill(p, 1000)
threaded_out = tmap(p_val -> compute_trajectory_mean(@SVector([1.0,0.0,0.0]),p_val), test_ps)