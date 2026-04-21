"""
`solve_system(f,u0,n)`

Solves the dynamical system

``u_{n+1} = f(u_n)``

for N steps. Returns the solution at step `n` with parameters `p`.

"""

# =============================================================================
# 1. 最佳性能实现：原地预分配 + StaticArrays + @inbounds (3.775 μs, 0 allocations)
# =============================================================================
function solve_system_save!(u,f,u0,p,n)
    # 原地操作，完全零分配
    @inbounds u[1] = u0
    @inbounds for i in 1:length(u)-1
        u[i+1] = f(u[i],p)
    end
    u
end

# =============================================================================
# 2. 推荐实现：StaticArrays 版本 (4.643 μs, 3 allocations)
# =============================================================================
using StaticArrays

function lorenz_static(u,p)
    α,σ,ρ,β = p
    @inbounds begin
        du1 = u[1] + α*(σ*(u[2]-u[1]))
        du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
        du3 = u[3] + α*(u[1]*u[2] - β*u[3])
    end
    @SVector [du1,du2,du3]
end

function solve_system_save_static(f,u0,p,n)
    # 非原地操作，但使用StaticArrays优化
    u = Vector{typeof(u0)}(undef,n)
    @inbounds u[1] = u0
    @inbounds for i in 1:n-1
        u[i+1] = f(u[i],p)
    end
    u
end

# =============================================================================
# 3. 推荐实现：原地操作版本 (4.657 μs, 6 allocations)
# =============================================================================
function lorenz_mutate(du,u,p)
    α,σ,ρ,β = p
    du[1] = u[1] + α*(σ*(u[2]-u[1]))
    du[2] = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du[3] = u[3] + α*(u[1]*u[2] - β*u[3])
end

function solve_system_mutate(f,u0,p,n)
    # 创建工作缓冲区，通过交换指针实现高效原地操作
    du = similar(u0); u = copy(u0)
    for i in 1:n-1
        f(du,u,p)
        u,du = du,u  # 交换指针，避免复制
    end
    u
end

# =============================================================================
# 4. 基础版本：只返回最终结果 (7.767 μs, 2000 allocations)
# =============================================================================
function solve_system(f,u0,p,n)
    u = u0
    for i in 1:n-1
        u = f(u,p)
    end
    u
end

function lorenz_basic(u,p)
    α,σ,ρ,β = p
    du1 = u[1] + α*(σ*(u[2]-u[1]))
    du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du3 = u[3] + α*(u[1]*u[2] - β*u[3])
    [du1,du2,du3]
end

# =============================================================================
# 5. 一般性能：矩阵视图版本 (11.900 μs, 2003 allocations)
# =============================================================================
function solve_system_save_matrix_view(f,u0,p,n)
    u = Matrix{eltype(u0)}(undef,length(u0),n)
    u[:,1] = u0
    for i in 1:n-1
        u[:,i+1] = f(@view(u[:,i]),p)  # 使用视图避免复制
    end
    u
end

# =============================================================================
# 6. 不推荐：预分配矩阵版本 (19.400 μs, 4001 allocations)
# =============================================================================
function solve_system_save_matrix(f,u0,p,n)
    u = Matrix{eltype(u0)}(undef,length(u0),n)
    u[:,1] = u0
    for i in 1:n-1
        u[:,i+1] = f(u[:,i],p)
    end
    u
end

# =============================================================================
# 7. 绝对避免：动态扩容矩阵版本 (731.100 μs, 4917 allocations)
# =============================================================================
function solve_system_save_matrix_resize(f,u0,p,n)
    u = Matrix{eltype(u0)}(undef,length(u0),1)
    u[:,1] = u0
    for i in 1:n-1
        u = hcat(u,f(@view(u[:,i]),p))  # 每次hcat都要复制整个矩阵！
    end
    u
end

# =============================================================================
# 8. 动态数组版本（与预分配版本性能相近）
# =============================================================================
function solve_system_save_push(f,u0,p,n)
    u = Vector{typeof(u0)}(undef,1)
    u[1] = u0
    for i in 1:n-1
        push!(u,f(u[i],p))
    end
    u
end

# =============================================================================
# 错误示例：浅复制问题（所有结果都相同！）
# =============================================================================
function solve_system_save_shallow_copy_bug(f,u0,p,n)
    u = Vector{typeof(u0)}(undef,n)
    du = similar(u0)
    u[1] = u0
    for i in 1:n-1
        f(du,u[i],p)
        u[i+1] = du  # 浅复制！所有元素指向同一个数组
    end
    u
end

# =============================================================================
# 正确的深复制版本（但性能不如指针交换）
# =============================================================================
function solve_system_save_deep_copy(f,u0,p,n)
    u = Vector{typeof(u0)}(undef,n)
    du = similar(u0)
    u[1] = u0
    for i in 1:n-1
        f(du,u[i],p)
        u[i+1] = copy(du)  # 深复制，但有额外开销
    end
    u
end

# =============================================================================
# 性能测试和演示
# =============================================================================
using Plots
using BenchmarkTools

# 测试参数
p = (0.02,10.0,28.0,8/3)
u0_vector = [1.0,0.0,0.0]
u0_svector = @SVector[1.0,0.0,0.0]
n = 1000

println("=== 性能对比测试 ===")

# 1. 最佳性能实现
println("\n1. 原地预分配 + StaticArrays + @inbounds:")
u_buffer = Vector{typeof(u0_svector)}(undef, n)
@btime solve_system_save!(u_buffer, lorenz_static, u0_svector, p, n)

# 2. StaticArrays 版本
println("\n2. StaticArrays 版本:")
@btime solve_system_save_static(lorenz_static, u0_svector, p, n)

# 3. 原地操作版本
println("\n3. 原地操作版本:")
@btime solve_system_mutate(lorenz_mutate, u0_vector, p, n)

# 4. 基础版本
println("\n4. 基础版本:")
@btime solve_system(lorenz_basic, u0_vector, p, n)

# 5. 矩阵视图版本
println("\n5. 矩阵视图版本:")
@btime solve_system_save_matrix_view(lorenz_basic, u0_vector, p, n)

# 6. 预分配矩阵版本
println("\n6. 预分配矩阵版本:")
@btime solve_system_save_matrix(lorenz_basic, u0_vector, p, n)

# 7. 动态扩容矩阵版本（注释掉以避免长时间等待）
# println("\n7. 动态扩容矩阵版本:")
# @btime solve_system_save_matrix_resize(lorenz_basic, u0_vector, p, 100)  # 减少步数

# 8. 动态数组版本
println("\n8. 动态数组版本:")
@btime solve_system_save_push(lorenz_basic, u0_vector, p, n)

println("\n=== 可视化演示 ===")
# 使用最佳实现进行可视化
to_plot = solve_system_save_static(lorenz_static, u0_svector, p, 1000)
x = [to_plot[i][1] for i in 1:length(to_plot)]
y = [to_plot[i][2] for i in 1:length(to_plot)]
z = [to_plot[i][3] for i in 1:length(to_plot)]
plot(x,y,z, title="Lorenz Attractor (Optimized Implementation)")

println("\n=== 关键优化原则总结 ===")
println("1. 避免不必要的内存分配：预分配 > 动态扩容")
println("2. 使用合适的数据结构：StaticArrays 对小的固定大小数组更优")
println("3. 原地操作：通过缓冲区交换减少分配")
println("4. 消除边界检查：在确保安全的情况下使用 @inbounds")
println("5. 避免矩阵的频繁复制：向量存储通常比矩阵更高效")


