# =============================================================================
# Lesson 1: Julia 性能优化基础
# =============================================================================

using BenchmarkTools

# =============================================================================
# 1. 内存分配 vs 预分配 (Memory Allocation vs Pre-allocation)
# =============================================================================
println("=== 1. 内存分配 vs 预分配 ===")

A = rand(100, 100) 
B = rand(100, 100) 
C = zeros(100, 100)

# 预分配版本 - 推荐
function inner_noalloc!(C, A, B) 
    for j in 1:100, i in 1:100 
        val = A[i,j] + B[i,j] 
        C[i,j] = val
    end 
end 

# 动态分配版本 - 不推荐
function inner_alloc(A, B) 
    C = similar(A) 
    for j in 1:100, i in 1:100 
        val = A[i,j] + B[i,j] 
        C[i,j] = val
    end 
    return C
end 

println("预分配版本:")
@btime inner_noalloc!($C, $A, $B)
println("动态分配版本:")
@btime inner_alloc($A, $B)

# =============================================================================
# 2. 多重分派 (Multiple Dispatch)
# =============================================================================
println("\n=== 2. 多重分派 ===")

# 基于类型的函数重载
ff(x::Int, y::Int) = 2x + y 
ff(x::Float64, y::Float64) = x / y 
ff(x::Number, y::Number) = x + y  # 更通用的方法

@show ff(2, 5)      # 调用 Int 版本
@show ff(2.0, 5.0)  # 调用 Float64 版本
@show ff(2.0, 5)    # 调用 Number 版本

println("查看方法定义:")
@which +(2.0, 5)

# =============================================================================
# 3. 类型稳定性 (Type Stability)
# =============================================================================
println("\n=== 3. 类型稳定性 ===")

# 检查类型稳定性
a = [1.0, 2.0, 3.0]
function bad_container(a) 
    a[2] 
end 
println("检查 bad_container 的类型稳定性:")
@code_warntype bad_container(a)

# isbits 类型示例
isbits(1.0)
struct MyComplex 
    real::Float64 
    imag::Float64 
end 
println("MyComplex 是 isbits 类型:", isbits(MyComplex(1.0, 1.0)))

# 为自定义类型定义运算符
Base.:+(a::MyComplex, b::MyComplex) = MyComplex(a.real + b.real, a.imag + b.imag) 
Base.:+(a::MyComplex, b::Int) = MyComplex(a.real + b, a.imag) 
Base.:+(b::Int, a::MyComplex) = MyComplex(a.real + b, a.imag) 

# =============================================================================
# 4. 函数屏障 (Function Barrier) - 解决类型不稳定问题
# =============================================================================
println("\n=== 4. 函数屏障 ===")

# 类型不稳定的容器
x = Number[1.5, 2.5] 

# 方法 1：慢速 - 直接在循环中访问类型不稳定的数组
function slow_r(x)
    s = 0.0
    for i in 1:100
        # 编译器无法确定 x[1] 的类型，必须生成保守的机器码
        s += x[1] 
    end
    return s
end

# 方法 2：快速 - 使用函数屏障
function fast_s(x)
    # 在这里提取值，让编译器知道确切类型
    return _inner_kernel(x[1], x[2]) 
end

function _inner_kernel(x1, x2)
    # 进入此函数后，x1 和 x2 的类型是稳定的
    s = 0.0
    for i in 1:100
        s += x1
    end
    return s
end

println("慢速版本:")
@btime slow_r($x)
println("快速版本:")
@btime fast_s($x)
println("内核函数直接调用:")
@btime _inner_kernel($(x[1]), $(x[2]))

# =============================================================================
# 5. 内联优化 (Inlining)
# =============================================================================
println("\n=== 5. 内联优化 ===")

@noinline fnoinline(x, y) = x + y
finline(x, y) = x + y  # 自动内联

function qinline(x, y)
    a = 4
    b = 2
    c = finline(x, a)
    d = finline(b, c)
    finline(d, y)
end

function qnoinline(x, y)
    a = 4
    b = 2
    c = fnoinline(x, a)
    d = fnoinline(b, c)
    fnoinline(d, y)
end

println("内联版本:")
@btime qinline(1.0, 2.0)
println("非内联版本:")
@btime qnoinline(1.0, 2.0)

# =============================================================================
# 6. 边界检查优化 (@inbounds)
# =============================================================================
println("\n=== 6. 边界检查优化 ===")

function inner_noalloc_ib!(C, A, B)
    @inbounds for j in 1:100, i in 1:100
        val = A[i,j] + B[i,j]
        C[i,j] = val
    end
end

println("有边界检查:")
@btime inner_noalloc!($C, $A, $B)
println("无边界检查 (@inbounds):")
@btime inner_noalloc_ib!($C, $A, $B)

# =============================================================================
# 7. 数据结构选择：Vector{Array} vs Matrix
# =============================================================================
println("\n=== 7. 数据结构选择 ===")
println("""
Vector{Array} vs. Matrix 对比:

特性                | Matrix (2D Array)           | Vector{Vector} (Jagged)
--------------------|-----------------------------|------------------------
内存布局            | 连续存储，CPU缓存命中率高   | 不连续，指针跳转开销大
灵活性              | 固定矩形形状                | 每行可不同长度（锯齿状）
性能（速度）        | 极快，支持SIMD向量化        | 较慢，难以SIMD优化
改变大小            | 昂贵，通常需重新分配        | 便宜，可单独修改某行
""")

# =============================================================================
# 8. 泛型函数的优势与注意事项
# =============================================================================
println("\n=== 8. 泛型函数 ===")
println("""
泛型函数的优势：
✅ 自动特化：针对不同类型生成最优机器码
✅ 无限组合性：不同类型可无缝组合使用
✅ 代码复用：一套逻辑处理多种数据类型

需要注意的问题：
⚠️ 方法歧义：多个方法签名可能导致调用冲突
⚠️ 代码膨胀：过多类型特化导致二进制文件变大
⚠️ 类型陷阱：确保返回类型只依赖输入类型，不依赖输入值
""")

# =============================================================================
# 9. 数据框实现建议
# =============================================================================
println("\n=== 9. 数据框实现 ===")
println("""
在Julia中实现高性能数据框的核心思路：列存储 (Columnar Storage)
推荐方案：NamedTuple of Vectors
- 每列独立存储，类型稳定
- 内存连续，缓存友好
- 支持高效的列操作
""")