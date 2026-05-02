function f(A, x)
    i = length(A)

    @label Loop
    a = A[i]    # Load
    c = a + x   # Add
    A[i] = c    # Store
    i = i - 1   # Decrement
    i > 0 && @goto Loop

    return A
end

using SIMD
A = rand(Float64, 64)
T = Vec{4, Float64}
x = 1.0

for i in 1:4:length(A)
    a = vload(T, A, i)
    c = a + x
    vstore(c, A, i)
end

A = rand(Int64, 64)
for i in 1:length(A)
    a = A[i]
    if a % 2 == 0
        A[i] = -a
    end
end

using SIMD
A = rand(Int64, 64)
T = Vec{4, Int64}

for i in 1:4:length(A)
    a = vload(T, A, i)
    mask = a % 2 == 0        # calculate mask
    b = -a                   # If branch
    c = vifelse(mask, b, a)  # merge results
    vstore(c, A, i)
end

m = Chain(
  Conv((2,2), 1=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax) |> gpu
  