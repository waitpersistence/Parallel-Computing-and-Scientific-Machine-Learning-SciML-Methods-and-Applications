# Lesson 11: 自动微分 (Automatic Differentiation) 实现示例
# 本文件展示了多种自动微分的实现方式，包括前向模式和反向模式

# =============================================================================
# 基础数据结构定义
# =============================================================================

struct Call{F,As<:Tuple}
  func::F
  args::As
end

mutable struct Tracked{T}
  ref::UInt32
  f::Call
  isleaf::Bool
  grad::T
  Tracked{T}(f::Call) where T = new(0, f, false)
  Tracked{T}(f::Call, grad::T) where T = new(0, f, false, grad)
  Tracked{T}(f::Call{Nothing}, grad::T) where T = new(0, f, true, grad)
end

mutable struct TrackedReal{T<:Real} <: Real
  data::T
  tracker::Tracked{T}
end

struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  tracker::Tracked{A}
  data::A
  grad::A
  TrackedArray{T,N,A}(t::Tracked{A}, data::A) where {T,N,A} = new(t, data)
  TrackedArray{T,N,A}(t::Tracked{A}, data::A, grad::A) where {T,N,A} = new(t, data, grad)
end

# =============================================================================
# 基础函数：计算平方根（牛顿法）
# =============================================================================

"""
计算 x 的平方根，使用牛顿迭代法
"""
function f(x)
    a = x
    for i in 1:300
        a = 0.5 * (a + x/a)
    end
    a
end

"""
分解版本的平方根计算，显示中间步骤
"""
function f_lowered(x)
    a = x
    for i in 1:300
        tmp1 = x / a
        tmp2 = a + tmp1
        a = 0.5 * tmp2
    end
    y = a
end

# =============================================================================
# 前向模式自动微分
# =============================================================================

"""
前向模式自动微分实现
"""
function f_forward(x, dx)
    a, da = (x, dx)
    for i in 1:300
        tmp1, dtmp1 = (x / a, (dx * a - da * x) / a^2)
        tmp2, dtmp2 = (a + tmp1, da + dtmp1)
        a, da = (0.5 * tmp2, 0.5 * dtmp2)
    end
    y, dy = (a, da)
end

# 测试前向模式
f_forward(2.0, 1.0) 

# =============================================================================
# 反向模式自动微分 - 拉回函数 (Pullback Functions)
# =============================================================================

function apply_pullback(::typeof(identity), args, out, outbar)
    (outbar,)
end

function apply_pullback(::typeof(/), args, out, outbar)
    a, b = args
    (outbar / b, -outbar * a / b^2)
end

function apply_pullback(::typeof(+), args, out, outbar)
    (outbar, outbar)
end

function apply_pullback(::typeof(*), args, out, outbar)
    a, b = args
    (outbar * b, outbar * a)
end

# =============================================================================
# 反向模式自动微分 - 动态图实现 (Tape-based)
# =============================================================================

"""
动态图反向模式自动微分实现
使用 tape 记录计算过程，然后反向传播梯度
"""
function f_reverse_dynamic_ad(x)
    node_id = 0
    next_id() = (node_id += 1; node_id)

    tape = []  # entries: (op, input_ids, output_id)
    is_constant = Set{Int}()

    x_id = next_id()

    a_id = next_id()
    push!(tape, (identity, (x_id,), a_id))

    const_05_id = next_id()
    push!(is_constant, const_05_id)

    for i in 1:300
        tmp1_id = next_id()
        push!(tape, (/, (x_id, a_id), tmp1_id))

        tmp2_id = next_id()
        push!(tape, (+, (a_id, tmp1_id), tmp2_id))

        a_id = next_id()
        push!(tape, (*, (const_05_id, tmp2_id), a_id))
    end

    y_id = a_id

    # Replay tape to get forward values
    node_vals = Dict{Int,Float64}()
    node_vals[x_id] = x
    node_vals[const_05_id] = 0.5

    for (op, in_ids, out_id) in tape
        args = ntuple(j -> node_vals[in_ids[j]], length(in_ids))
        node_vals[out_id] = op === identity ? args[1] : op(args...)
    end

    y = node_vals[y_id]

    function reversepass(ybar)
        adj = Dict{Int,Float64}()
        adj[y_id] = ybar

        for i in length(tape):-1:1
            op, in_ids, out_id = tape[i]
            outbar = get(adj, out_id, 0.0)
            args = ntuple(j -> node_vals[in_ids[j]], length(in_ids))
            bars = apply_pullback(op, args, node_vals[out_id], outbar)
            for (j, id) in enumerate(in_ids)
                if id ∉ is_constant
                    adj[id] = get(adj, id, 0.0) + bars[j]
                end
            end
        end
        get(adj, x_id, 0.0)
    end

    y, reversepass
end

# 测试动态图反向模式
y, pullback = f_reverse_dynamic_ad(2.0)
(y, pullback(1.0))

# =============================================================================
# 反向模式自动微分 - 基于内存存储的实现
# =============================================================================

"""
反向模式自动微分 - 存储所有中间值
"""
function f_reverse_memory(x, da)
    # Forward pass: store all intermediate a values
    as = zeros(301)
    as[1] = x
    for i in 1:300
        tmp1 = x / as[i]
        tmp2 = as[i] + tmp1
        as[i+1] = 0.5 * tmp2
    end

    y = as[end]

    # Reverse pass
    abar = da
    xbar = 0.0

    for i in 300:-1:1
        # Reverse of: a[i+1] = 0.5 * tmp2
        tmp2bar = abar * 0.5

        # Reverse of: tmp2 = a[i] + tmp1
        tmp1bar = tmp2bar
        abar_from_add = tmp2bar

        # Reverse of: tmp1 = x / a[i]
        xbar += tmp1bar / as[i]
        abar_from_div = tmp1bar * (-x / as[i]^2)

        # Total adjoint for a[i]
        abar = abar_from_add + abar_from_div
    end

    # Reverse of initial: a = x
    xbar += abar

    y, xbar
end

# =============================================================================
# 反向模式自动微分 - 无内存存储的实现
# =============================================================================

"""
反向模式自动微分 - 不存储中间值，通过逆运算重构
"""
function f_reverse_memoryless(x, da)
    a = x
    for i in 1:300
        tmp1 = x / a
        tmp2 = a + tmp1
        a = 0.5 * tmp2
    end

    aout = a

    # Reverse pass: reconstruct intermediates by inverting each step
    abar = da
    xbar = 0.0

    for i in 300:-1:1
        # Invert: a_new = 0.5 * tmp2 => tmp2 = 2 * a
        tmp2 = 2 * a
        # Invert: a_old^2 - tmp2*a_old + x = 0 => take larger root
        a_old = (tmp2 + sqrt(abs(tmp2^2 - 4x))) / 2
        tmp1 = x / a_old

        tmp2bar = abar * 0.5
        tmp1bar = tmp2bar
        abar_from_add = tmp2bar

        xbar += tmp1bar / a_old
        abar_from_div = tmp1bar * (-x / a_old^2)

        abar = abar_from_add + abar_from_div
        a = a_old
    end

    xbar += abar

    aout, xbar
end

# =============================================================================
# 基于 while 循环的实现
# =============================================================================

"""
使用 while 循环计算平方根（自适应迭代次数）
"""
function f_while(x)
    a = x
    while abs(a - x/a) > 1e-14
        a = 0.5 * (a + x/a)
    end
    a
end

"""
while 循环版本的反向模式 - 基于内存存储
"""
function f_while_reverse_memory(x, da)
    as = [x]
    while abs(as[end] - x / as[end]) > 1e-14
        a_prev = as[end]
        tmp1 = x / a_prev
        tmp2 = a_prev + tmp1
        push!(as, 0.5 * tmp2)
    end

    y = as[end]
    abar = da
    xbar = 0.0

    for i in (length(as)):-1:2
        tmp2bar = abar * 0.5
        tmp1bar = tmp2bar
        abar_from_add = tmp2bar

        xbar += tmp1bar / as[i-1]
        abar_from_div = tmp1bar * (-x / as[i-1]^2)

        abar = abar_from_add + abar_from_div
    end

    xbar += abar
    y, xbar
end

"""
while 循环版本的反向模式 - 无内存存储
"""
function f_while_reverse_memoryless(x, da)
    a = x
    iters = 0
    while abs(a - x/a) > 1e-14
        iters += 1
        tmp1 = x / a
        tmp2 = a + tmp1
        a = 0.5 * tmp2
    end

    aout = a
    abar = da
    xbar = 0.0

    for i in iters:-1:1
        tmp2 = 2 * a
        a_old = (tmp2 + sqrt(abs(tmp2^2 - 4x))) / 2
        tmp1 = x / a_old

        tmp2bar = abar * 0.5
        tmp1bar = tmp2bar
        abar_from_add = tmp2bar

        xbar += tmp1bar / a_old
        abar_from_div = tmp1bar * (-x / a_old^2)

        abar = abar_from_add + abar_from_div
        a = a_old
    end

    xbar += abar
    aout, xbar
end

# 测试 while 循环版本
f_while_reverse_memory(2.0, 1.0)
f_while_reverse_memoryless(2.0, 1.0)

# =============================================================================
# 结果比较和验证
# =============================================================================

# 计算精确导数值（sqrt(x) 的导数是 1/(2*sqrt(x))）
x = 2.0
exact = 1 / (2*sqrt(x))

# 计算各种方法的结果
_, d_fwd = f_forward(x, 1.0)
_, d_mem = f_reverse_memory(x, 1.0)
_, d_mless = f_reverse_memoryless(x, 1.0)
_, pb = f_reverse_dynamic_ad(x)
d_dyn = pb(1.0)
_, d_wmem = f_while_reverse_memory(x, 1.0)
_, d_wmless = f_while_reverse_memoryless(x, 1.0)

# 输出结果比较
println("Exact derivative:        $exact")
println("Forward-mode:            $d_fwd    (err = $(abs(d_fwd - exact)))")
println("Reverse dynamic (tape):  $d_dyn    (err = $(abs(d_dyn - exact)))")
println("Reverse memory:          $d_mem    (err = $(abs(d_mem - exact)))")
println("Reverse memoryless:      $d_mless  (err = $(abs(d_mless - exact)))")
println("While + memory:          $d_wmem   (err = $(abs(d_wmem - exact)))")
println("While + memoryless:      $d_wmless (err = $(abs(d_wmless - exact)))")