using OrdinaryDiffEq, Measurements
gaccel = 9.79 ± 0.02; # Gravitational constants
L = 1.00 ± 0.01; # Length of the pendulum

#Initial Conditions
u₀ = [0 ± 0, π / 3 ± 0.02] # Initial speed and initial angle
tspan = (0.0, 6.3)

#Define the problem
function simplependulum(du,u,p,t)
    θ  = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(gaccel/L) * sin(θ)
end

#Pass to solvers
prob = ODEProblem(simplependulum, u₀, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-6)

using Plots
plot(sol,plotdensity=100,idxs=2)


gaccel = interval(9.77,9.81); # Gravitational constants
L = interval(0.99,1.01); # Length of the pendulum

#Initial Conditions
u₀ = [interval(0,0), interval(((π / 3.0)-0.02),((π / 3.0)+0.02))] # Initial speed and initial angle
tspan = (0.0, 6.3)

#Define the problem
function simplependulum(du,u,p,t)
    θ  = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = -(gaccel/L) * sin(θ)
end

#Pass to solvers
prob = ODEProblem(simplependulum, u₀, tspan)
sol = solve(prob, Euler(), dt=0.1)