using Lux
using Random
using ComponentArrays
using OrdinaryDiffEq
using Optimization
using OptimizationOptimisers
using Plots
using NeuralPDE
using Flux
using Test

u0 = Float32[0.8; 0.8]
tspan = (0.0f0,25.0f0)

ann = Chain(Dense(2,10,tanh), Dense(10,1))

p_nn, st = Lux.setup(Xoshiro(0), ann)
p_nn = ComponentArray(p_nn)
p_ode = Float32[-2.0,1.1]
p3 = ComponentArray(nn=p_nn, ode=p_ode)

function dudt_(du,u,p,t)
    x, y = u
    nn_out, _ = ann(u, p.nn, st)
    du[1] = nn_out[1]
    du[2] = p.ode[1]*y + p.ode[2]*x
end
prob = ODEProblem(dudt_,u0,tspan,p3)
solve(prob,Tsit5(),abstol=1e-8,reltol=1e-6)

function predict_adjoint(p)
  Array(solve(remake(prob,p=p,u0=u0),Tsit5(),saveat=0.0:0.1:25.0))
end
loss_adjoint(p) = sum(abs2,x-1 for x in predict_adjoint(p))
loss_adjoint(p3)

iter = 0
cb = function (p, l)
  global iter += 1
  if iter % 50 == 0
    display(l)
    display(plot(solve(remake(prob,p=p,u0=u0),Tsit5(),saveat=0.1),ylim=(0,6)))
  end
  return false
end

# Display the ODE with the current parameter values.
cb(p3, loss_adjoint(p3))

optprob = Optimization.OptimizationProblem((p, _) -> loss_adjoint(p), p3)
res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback = cb, maxiters = 300)

# Nonlinear Black-Scholes Equation with Default Risk
d = 100 # number of dimensions
x0 = fill(100.0f0,d)
tspan = (0.0f0,1.0f0)
dt = 0.125 # time step
m = 100 # number of trajectories (batch size)
time_steps = div(tspan[2]-tspan[1],dt)

g(X) = minimum(X)
δ = 2.0f0/3
R = 0.02f0
f(X,u,σᵀ∇u,p,t) = -(1 - δ)*Q(u)*u - R*u

vh = 50.0f0
vl = 70.0f0
γh = 0.2f0
γl = 0.02f0
function Q(u)
    Q = 0
    if u < vh
        Q = γh
    elseif  u >= vl
        Q = γl
    else  #if  u >= vh && u < vl
        Q = ((γh - γl) / (vh - vl)) * (u - vh) + γh
    end
end

µc = 0.02f0
σc = 0.2f0

μ(X,p,t) = µc*X #Vector d x 1
σ(X,p,t) = σc*Diagonal(X) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ, σ, x0, tspan)

hls = 10 + d #hidden layer size
opt = Flux.ADAM(0.008)  #optimizer
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))

σᵀ∇u = [Flux.Chain(Dense(d,hls,relu),
                   Dense(hls,hls,relu),
                   Dense(hls,hls,relu),
                   Dense(hls,d)) for i in 1:time_steps]
alg = NNPDEHan(u0, σᵀ∇u, opt = opt)

ans = solve(prob, alg, verbose = true, abstol=1e-8, maxiters = 100, dt=dt, trajectories=m)

prob_ans = 57.3 #60.781
error_l2 = sqrt((ans - prob_ans)^2/ans^2)

println("Nonlinear Black-Scholes Equation with Default Risk")
# println("numerical = ", ans)
# println("prob_ans = " , prob_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 0.1