using Sobol, Plots
seq = SobolSeq(2)
p = hcat([next!(seq) for i = 1:1024]...)'
scatter(p[:,1], p[:,2])

using LatinHypercubeSampling
p = LHCoptim(120,2,1000)
scatter(p[1][:,1],p[1][:,2])