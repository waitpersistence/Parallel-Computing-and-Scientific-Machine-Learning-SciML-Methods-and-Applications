using MPI
MPI.Init()
comm = MPI.COMM_WORLD
const MAX_N=25
const MAX_SIZE=2^MAX_N
# 两个进程都要分配，否则收发会没地方存
send_buffer = Vector{Int8}(undef, MAX_SIZE)
recv_buffer = Vector{Int8}(undef, MAX_SIZE)
# (可选) 给 send_buffer 填入一些随机数据，模拟真实载荷
send_buffer .= rand(Int8, MAX_SIZE)
rank = MPI.Comm_rank(comm)
print("I am rank $(MPI.Comm_rank(comm))\n")
for i in 1:25
    # 1. 同步：确保两个进程都准备好了
    MPI.Barrier(comm)
    len=2^i
    if rank ==0
        t_start = MPI.Wtime() # 记录发送开始时间
        # 2. 发送：发送数据给 rank=1
        MPI.Send(@view(send_buffer[1:len]), 1, i, comm)
        MPI.Recv!(recv_buffer, 1, i, comm)# 接收数据
        t_end = MPI.Wtime()
        T_i = (t_end - t_start) / 2
        bandwidth = (2^i) / T_i # 单位是 Bytes/s
        # 建议换算成 MiB/s: bandwidth / (1024^2)
        println("n=$i | Time: $(round(T_i, digits=6))s | BW: $(round(bandwidth/(1024^2), digits=2)) MiB/s")
    elseif rank ==1
        # 3. 接收：接收来自 rank=0 的数据
        MPI.Recv!(recv_buffer, 0, i, comm)
        MPI.Send(@view(recv_buffer[1:len]), 0, i, comm)
        # 4. 处理：处理接收到的数据
        #println("Process $rank received data: ", recv_buffer[1:len])
    end
end
MPI.Finalize()

n_vals = 2:25  # 记得跳过 n=1
# 将你刚才得到的 BW 数据填入
bw_vals = [0.29, 0.82, 2.37, 24.41, 26.54, 38.75,53.07,56.45,98.18,189.62,182.96,656.51,860.88,2682.4,2637.13
,2883.51,5543.24,5834.31,6731.74,6535.95,6709.72,5792.9,5666.53, 5882.95]
using Plots
gr() # 使用 GR 后端

plot(n_vals, bw_vals, 
    title = "MPI Bandwidth vs Message Size",
    xlabel = "Message Size (2^n bytes)",
    ylabel = "Bandwidth (MiB/s)",
    marker = (:circle, 4),      # 增加数据点标记
    line = (:steppre, 1),       # 或者用实线
    label = "Shared Memory MPI",
    grid = true)

# 建议保存为图片用于提交作业
savefig("bandwidth_plot.png")