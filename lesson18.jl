struct MyStruct
  a::AbstractArray
end
x = MyStruct([1,2,3])
function f(x)
  x.a[1]
end
using InteractiveUtils
@code_warntype f(x)