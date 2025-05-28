## StragglAR
StragglAR is a novel AllReduce algorithm for multi-GPU environments with a persistent straggler.
Specifically, the multi-GPU environment should have a homogeneous topology and enable any-to-any 
connectivity, such as today's popular NVSwitched multi-GPU servers. StragglAR can outperform 
Ring AllReduce, both theoretically (in communication complexity) and practically, 
in these environments when there is a persistent straggler GPU.

## Usage
The following instructions are for files in the `code/` subdirectory. Other subdirectories like `motivation/` (Section 2 experiments, to come) and `implementation/` (end-to-end experiments for distributed ML training, in Appendix F) have their own READMEs.

To synthesize schedules for power-of-2 number of GPUs (n):
- Run: `python synthesizer_pow2.py <n>` where n is the number of GPUs

To synthesize schedules for a non-power-of-2, even number of GPUs (n):
- Run: `python synthesizer_nonpow2.py <n>` where n is the number of GPUs

To compile AllReduce for 4 GPUs:
- Compile AllReduce: `nvcc -diag-suppress=177 -I${NCCL_HOME}/include -L${NCCL_HOME}/lib -lnccl  -o allreduce allreduce_4GPU.cu -std=c++17`

To compile AllReduce for 8 GPUs:
- Compile AllReduce: `nvcc -diag-suppress=177 -I${NCCL_HOME}/include -L${NCCL_HOME}/lib -lnccl  -o allreduce allreduce_8GPU.cu -std=c++17`

To run AllReduce: 
- `./allreduce <NUM_BYTES> <ALG> <ITERS> <DELAY>` where NUM_BYTES is the buffer size in bytes, ALG is one of ['stragglar', 'ring', 'rhd', 'direct'] (i.e., [StragglAR, Ring, Recursive Halving and Doubling, Direct]), and delay is the straggler delay time in ms (-1 means we ignore the concept of delay and assume the pre-work, either ReduceScatter for StragglAR or AllReduce for Direct, has already completed)

To obtain the simulation results using the analytical model:
- Run: `python simulation.py`
