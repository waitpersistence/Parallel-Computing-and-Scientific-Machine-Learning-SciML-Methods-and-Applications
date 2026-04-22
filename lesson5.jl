struct MyComplex
  real::Float64
  imag::Float64
end
arr = [MyComplex(rand(),rand()) for i in 1:100]

struct MyComplexes
  real::Vector{Float64}
  imag::Vector{Float64}
end
arr2 = MyComplexes(rand(100),rand(100))

using InteractiveUtils
Base.:+(x::MyComplex,y::MyComplex) = MyComplex(x.real+y.real,x.imag+y.imag)
Base.:/(x::MyComplex,y::Int) = MyComplex(x.real/y,x.imag/y)
average(x::Vector{MyComplex}) = sum(x)/length(x)
@code_llvm average(arr)

using InteractiveUtils
Base.:+(x::MyComplex,y::MyComplex) = MyComplex(x.real+y.real,x.imag+y.imag)
Base.:/(x::MyComplex,y::Int) = MyComplex(x.real/y,x.imag/y)
average(x::Vector{MyComplex}) = sum(x)/length(x)
@code_llvm average(arr)

using SIMD
v = Vec{4,Float64}((1,2,3,4))
@show v+v # basic arithmetic is supported
@show sum(v) # basic reductions are supported

using Base.Threads
Threads.nthreads() # should not be 1

using BenchmarkTools
acc = 0
@threads for i in 1:10_000
    global acc
    acc += 1
end
acc

const a2 = zeros(nthreads()*10)
const acc_lock2 = Ref{Int64}(0)
const splock2 = SpinLock()
function f_order()
    @threads for i in 1:length(a2)
        lock(splock2)
        acc_lock2[] += 1
        a2[i] = acc_lock2[]
        unlock(splock2)
    end
end
f_order()
a2


using CUDA

N = 2^20
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads=256 gpu_add2!(y_d, x_d)
all(Array(y_d) .== 3.0f0)

using Dagger

add1(value) = value + 1
add2(value) = value + 2
combine(a...) = sum(a)

p = delayed(add1)(4)
q = delayed(add2)(p)
r = delayed(add1)(3)
s = delayed(combine)(p, q, r)

@assert collect(s) == 16