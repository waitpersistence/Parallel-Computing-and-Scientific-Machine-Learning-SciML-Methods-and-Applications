# Lesson 8: Automatic Differentiation (自动微分)
# ==================================================

@show eps(Float64)
@show sqrt(eps(Float64))

using InteractiveUtils

# ==================================================
# Part 1: Dual Numbers Implementation (对偶数实现)
# ==================================================

"""
Dual数结构体，用于前向模式自动微分
- val: 函数值
- der: 导数值
"""
struct Dual{T}
    val::T   # value
    der::T  # derivative
end

# 基本运算符重载
# ----------------

# 加法
Base.:+(f::Dual, g::Dual) = Dual(f.val + g.val, f.der + g.der)
Base.:+(f::Dual, α::Number) = Dual(f.val + α, f.der)
Base.:+(α::Number, f::Dual) = f + α

# 减法
Base.:-(f::Dual, g::Dual) = Dual(f.val - g.val, f.der - g.der)

# 乘法 (乘积法则)
Base.:*(f::Dual, g::Dual) = Dual(f.val*g.val, f.der*g.val + f.val*g.der)
Base.:*(α::Number, f::Dual) = Dual(f.val * α, f.der * α)
Base.:*(f::Dual, α::Number) = α * f

# 除法 (商法则)
Base.:/(f::Dual, g::Dual) = Dual(f.val/g.val, (f.der*g.val - f.val*g.der)/(g.val^2))
Base.:/(α::Number, f::Dual) = Dual(α/f.val, -α*f.der/f.val^2)
Base.:/(f::Dual, α::Number) = f * inv(α)

# 幂运算
Base.:^(f::Dual, n::Integer) = Base.power_by_squaring(f, n)

# 指数函数
import Base: exp
exp(f::Dual) = Dual(exp(f.val), exp(f.val) * f.der)

# ==================================================
# Part 2: Basic Usage Examples (基本使用示例)
# ==================================================

# 创建Dual数实例
fd = Dual(3, 4)
gd = Dual(5, 6)

# 测试加法
fd + gd

# 辅助函数用于性能测试
add(a1, a2, b1, b2) = (a1+b1, a2+b2)
add(1, 2, 3, 4)

# 性能基准测试
using BenchmarkTools
a, b, c, d = 1, 2, 3, 4
@btime add($(Ref(a))[], $(Ref(b))[], $(Ref(c))[], $(Ref(d))[])

a = Dual(1, 2)
b = Dual(3, 4)
add(j1, j2) = j1 + j2
add(a, b)
@btime add($(Ref(a))[], $(Ref(b))[])

# ==================================================
# Part 3: Automatic Differentiation Functions (自动微分函数)
# ==================================================

"""
计算单变量函数在点x处的导数
"""
derivative(f, x) = f(Dual(x, one(x))).der  

# 测试导数计算
hf(x) = x^2 + 2
a = 3
xx = Dual(a, 1)
hf(xx)

# 计算多项式导数
derivative(x -> 3x^5 + 2, 2)

# 多变量函数的部分导数
fquad(x, y) = x^2 + x*y
a, b = 3.0, 4.0
fquad_1(x) = fquad(x, b)  # 固定y=b，对x求导
derivative(fquad_1, a)

# ==================================================
# Part 4: Multi-variable Dual Numbers (多变量对偶数)
# ==================================================

using StaticArrays

"""
多变量Dual数，支持N个变量的偏导数
"""
struct MultiDual{N,T}
    val::T
    derivs::SVector{N,T}
end

import Base: +, *

function +(f::MultiDual{N,T}, g::MultiDual{N,T}) where {N,T}
    return MultiDual{N,T}(f.val + g.val, f.derivs + g.derivs)
end

function *(f::MultiDual{N,T}, g::MultiDual{N,T}) where {N,T}
    return MultiDual{N,T}(f.val * g.val, f.val .* g.derivs + g.val .* f.derivs)
end

# 多变量函数示例
gcubic(x, y) = x*x*y + x + y
(a, b) = (1.0, 2.0)

# 创建多变量Dual数 (x对x的偏导为1，对y的偏导为0；y对x的偏导为0，对y的偏导为1)
xx = MultiDual(a, SVector(1.0, 0.0))
yy = MultiDual(b, SVector(0.0, 1.0))

gcubic(xx, yy)

# 向量值函数
fsvec(x, y) = SVector(x*x + y*y , x + y)
fsvec(xx, yy)

# ==================================================
# Part 5: Using ForwardDiff Library (使用ForwardDiff库)
# ==================================================

using ForwardDiff, StaticArrays

# 使用ForwardDiff计算梯度
ForwardDiff.gradient(xx -> ((x, y) = xx; x^2 * y + x*y), [1, 2])

# ==================================================
# Part 6: Newton's Method with Automatic Differentiation (牛顿法与自动微分)
# ==================================================

"""
牛顿法的单步迭代
"""
function newton_step(f, x0)
    J = ForwardDiff.jacobian(f, x0)
    δ = J \ f(x0)
    return x0 - δ
end

"""
牛顿法求解非线性方程组
"""
function newton(f, x0)
    x = x0
    for i in 1:10
        x = newton_step(f, x)
        @show x
    end
    return x
end

# 测试牛顿法：求解方程组
# x^2 + y^2 = 1
# x - y = 0
fsvec2(xx) = ((x, y) = xx; SVector(x^2 + y^2 - 1, x - y))
x0 = SVector(3.0, 5.0)
x = newton(fsvec2, x0)