#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda.h>

__global__ void reduce_add_kernel(float* __restrict__ dst,
                                  const float* __restrict__ src,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// thin C++ wrapper so the host can launch it
void launch_reduce_add(float* dst, const float* src, int n,
                       cudaStream_t stream) {
  if (n == 0) return;
  int threads = 128;
  int blocks  = (n + threads - 1) / threads;
  reduce_add_kernel<<<blocks, threads>>>(dst, src, n);
}