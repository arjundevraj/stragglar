#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <assert.h>
#include <cstring> 
#include <vector>
#include <utility>
#include <cstdint> 

#define RED_ADD_THREADS 256

#define CHECK_CUDA(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define CHECK_NCCL(cmd) do { \
  ncclResult_t res = cmd; \
  if (res != ncclSuccess) { \
    printf("NCCL error %s:%d: '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)


__global__ void fill_pattern(float* dst, float v, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += gridDim.x * blockDim.x)
      dst[i] = v;
}

int main(int argc, char* argv[]) {
  const int numRanks = 8;

  int version;
  ncclGetVersion(&version);

  // printf("NCCL version %d\n", version);
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <bufferSize> <numIters>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  size_t bytes = (size_t)strtoull(argv[1], NULL, 10);
  size_t size = bytes / sizeof(float);
  int numIters = atoi(argv[2]);

  int nGPUs = 0;
  CHECK_CUDA(cudaGetDeviceCount(&nGPUs));
  if (nGPUs < numRanks) {
    printf("Need at least %d GPUs\n", numRanks);
    return -1;
  }

  int devs[numRanks];
  for (int i = 0; i < numRanks; ++i) devs[i] = i;

  // Allocate device buffers
  float* d_buffers[numRanks];
  float* d_tempbufs[numRanks];
  cudaStream_t streams[numRanks];
  ncclComm_t subComms[numRanks - 1];

  cudaSetDevice(devs[0]);
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  
  CHECK_NCCL(ncclCommInitAll(subComms, numRanks - 1, NULL));

  size_t chunkSize = size / (numRanks - 1);

  for (int i = 0; i < numRanks; ++i) {
    CHECK_CUDA(cudaSetDevice(devs[i]));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaMallocAsync(&d_buffers[i], size * sizeof(float), streams[i]));
  }

  for (int iter = 0; iter < numIters + 1; ++iter) {
    // Reset buffers if needed (same init pattern as above)
    for (int i = 0; i < numRanks; ++i) {
      CHECK_CUDA(cudaSetDevice(devs[i]));
      fill_pattern<<<(size+255)/256, 256, 0, streams[i] >>>(d_buffers[i], float(i+1), size);
    }

    for (int i = 0; i < numRanks; ++i) {
      CHECK_CUDA(cudaSetDevice(devs[i]));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    size_t chunkSize = size / (numRanks - 1);
    cudaSetDevice(devs[0]);
    cudaEventRecord(start, streams[0]);
    for (int r = 0; r < numRanks - 1; ++r) {
      cudaSetDevice(devs[r]);
      cudaStreamSynchronize(streams[r]);
    }
  
    ncclGroupStart();
    for (int r = 0; r < numRanks - 1; ++r) {
      cudaSetDevice(devs[r]);
      ncclReduceScatter(d_buffers[r], d_buffers[r] + (r * chunkSize), chunkSize, ncclFloat, ncclSum, subComms[r], streams[r]);
    }
    ncclGroupEnd();
    cudaSetDevice(devs[0]);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms;
    cudaSetDevice(devs[0]);
    cudaEventElapsedTime(&ms, start, stop);
    if (iter == 0) continue;
    float bw = (float)size * sizeof(float) / 1024.0 / 1024.0 / 1024.0 * 1000.0 / ms;
    printf("%zu,%d,%.3f,%.3f\n",
      (size_t)size * sizeof(float),   // bytes, still a size_t
      iter,
      ms,
      bw);
  }

  for (int i = 0; i < numRanks; ++i) {
    cudaSetDevice(devs[i]);
    cudaFree(d_buffers[i]);
    cudaStreamDestroy(streams[i]);  // Streams are local to each device
    if (i < numRanks - 1)
      CHECK_NCCL(ncclCommDestroy(subComms[i]));
  }
  
  return 0;
}