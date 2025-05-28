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

__global__ void reduce_add(float* dst, float* src, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] += src[i];
  }
}

__global__ void gpu_sleep_kernel(clock_t sleep_cycles) {
  clock_t start = clock();
  while (clock() - start < sleep_cycles);
}

__global__ void fill_pattern(float* dst, float v, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < n; i += gridDim.x * blockDim.x)
    dst[i] = v;
}

clock_t calculate_sleep_cycles(float ms, int* devs) {
  cudaSetDevice(devs[7]);

  // Query clock rate (in kHz)
  int clockRate_kHz;
  cudaDeviceGetAttribute(&clockRate_kHz, cudaDevAttrClockRate, devs[3]);

  // Compute number of cycles to sleep
  clock_t sleep_cycles = static_cast<clock_t>(ms * clockRate_kHz);

  return sleep_cycles;
}

void direct_allreduce_helper(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size) {
  for (int r = 0; r < numRanks; ++r) {
    cudaSetDevice(devs[r]);
    cudaStreamSynchronize(streams[r]);
  }

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0], size, ncclFloat, 7, comms[0], streams[0]);
  ncclRecv(d_tempbufs[0], size, ncclFloat, 7, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[7], size, ncclFloat, 0, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], size, ncclFloat, 0, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupEnd();
  cudaSetDevice(devs[0]);
  reduce_add<<<(size + 128 - 1) / 128, 128, 0, streams[0]>>>(d_buffers[0], d_tempbufs[0], size);
  cudaSetDevice(devs[7]);
  reduce_add<<<(size + 128 - 1) / 128, 128, 0, streams[7]>>>(d_buffers[7], d_tempbufs[7], size);

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0], size, ncclFloat, 1, comms[0], streams[0]);
  ncclRecv(d_buffers[1], size, ncclFloat, 0, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[7], size, ncclFloat, 6, comms[7], streams[7]);
  ncclRecv(d_buffers[6], size, ncclFloat, 7, comms[6], streams[6]);
  ncclGroupEnd();
  ncclGroupEnd();

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0], size, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[2], size, ncclFloat, 0, comms[2], streams[2]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[1], size, ncclFloat, 3, comms[1], streams[1]);
  ncclRecv(d_buffers[3], size, ncclFloat, 1, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[6], size, ncclFloat, 4, comms[6], streams[6]);
  ncclRecv(d_buffers[4], size, ncclFloat, 6, comms[4], streams[4]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[7], size, ncclFloat, 5, comms[7], streams[7]);
  ncclRecv(d_buffers[5], size, ncclFloat, 7, comms[5], streams[5]);
  ncclGroupEnd();
  ncclGroupEnd();
  cudaSetDevice(devs[0]);
  cudaEventRecord(stop, streams[0]);
  cudaEventSynchronize(stop);
}

void direct_allreduce_delay(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, ncclComm_t* subComms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size, clock_t sleep_cycles) {
  cudaSetDevice(devs[0]);
  cudaEventRecord(start, streams[0]);

  // sleep the straggler
  cudaSetDevice(devs[7]);
  gpu_sleep_kernel<<<1, 1, 0, streams[7]>>>(sleep_cycles);

  // Synchronize to make sure everything is idle
  for (int r = 0; r < numRanks - 1; ++r) {
    cudaSetDevice(devs[r]);
    cudaStreamSynchronize(streams[r]);
  }

  ncclGroupStart();
  for (int r = 0; r < numRanks - 1; ++r) {
    cudaSetDevice(devs[r]);
    ncclAllReduce(d_buffers[r], d_buffers[r], size, ncclFloat, ncclSum, subComms[r], streams[r]);
  }
  ncclGroupEnd();

  direct_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size);
}

void direct_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size) {
  cudaSetDevice(devs[0]);
  cudaEventRecord(start, streams[0]);
  
  direct_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size);
}

void ring_allreduce_helper(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t chunkSize) {

  // Make sure everything is initialized
  for (int i = 0; i < numRanks; ++i) {
    CHECK_CUDA(cudaSetDevice(devs[i]));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));  // ensure idle
  }
    // Ring Reduce-Scatter
    for (int step = 1; step < numRanks; ++step) {
      ncclGroupStart();
      for (int r = 0; r < numRanks; ++r) {
        int sendTo = (r + 1) % numRanks;
        int recvFrom = (r - 1 + numRanks) % numRanks;
        int sendChunk = (r - step + numRanks) % numRanks;
        int recvChunk = (r - step - 1 + numRanks) % numRanks;

        float* sendPtr = d_buffers[r] + sendChunk * chunkSize;
        float* recvPtr = d_tempbufs[r];
        ncclGroupStart();
        ncclSend(sendPtr, chunkSize, ncclFloat, sendTo, comms[r], streams[r]);
        ncclRecv(recvPtr, chunkSize, ncclFloat, recvFrom, comms[r], streams[r]);
        ncclGroupEnd();
      }
      ncclGroupEnd();
      for (int r = 0; r < numRanks; ++r) {
        cudaSetDevice(devs[r]);
        int recvChunk = (r - step - 1 + numRanks) % numRanks;
        reduce_add<<<(chunkSize + RED_ADD_THREADS - 1) / RED_ADD_THREADS, RED_ADD_THREADS, 0, streams[r]>>>(d_buffers[r] + recvChunk * chunkSize, d_tempbufs[r], chunkSize);
      }
    }

    for (int step = 0; step < numRanks - 1; ++step) {
      ncclGroupStart();
      for (int r = 0; r < numRanks; ++r) {
        int sendTo = (r + 1) % numRanks;
        int recvFrom = (r - 1 + numRanks) % numRanks;
        int sendChunk = (r - step + numRanks) % numRanks;
        int recvChunk = (r - step - 1 + numRanks) % numRanks;

        float* sendPtr = d_buffers[r] + sendChunk * chunkSize;
        float* recvPtr = d_buffers[r] + recvChunk * chunkSize;
        ncclGroupStart();
        ncclSend(sendPtr, chunkSize, ncclFloat, sendTo, comms[r], streams[r]);
        ncclRecv(recvPtr, chunkSize, ncclFloat, recvFrom, comms[r], streams[r]);
        ncclGroupEnd();
      }
      ncclGroupEnd();
    }

    cudaSetDevice(devs[0]);
    cudaEventRecord(stop, streams[0]);
    cudaEventSynchronize(stop);
}


void ring_allreduce_delay(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size, clock_t sleep_cycles) {
  size_t chunkSize = size / numRanks;

  cudaSetDevice(devs[0]);
  cudaEventRecord(start, streams[0]);
  cudaSetDevice(devs[7]);
  gpu_sleep_kernel<<<1, 1, 0, streams[7]>>>(sleep_cycles);

  ring_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}

void ring_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size) {
  size_t chunkSize = size / numRanks;

  cudaSetDevice(devs[0]);
  cudaEventRecord(start, streams[0]);
  ring_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}


void rhd_allreduce_helper(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t chunkSize) {
  // Synchronize to make sure everything is idle
  for (int r = 0; r < numRanks; ++r) {
    cudaSetDevice(devs[r]);
    cudaStreamSynchronize(streams[r]);
  }

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0] + (4 * chunkSize), chunkSize * 4, ncclFloat, 1, comms[0], streams[0]);
  ncclRecv(d_tempbufs[0], chunkSize * 4, ncclFloat, 1, comms[0], streams[0]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[1], chunkSize * 4, ncclFloat, 0, comms[1], streams[1]);
  ncclRecv(d_tempbufs[1], chunkSize * 4, ncclFloat, 0, comms[1], streams[1]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[2] + (4 * chunkSize), chunkSize * 4, ncclFloat, 3, comms[2], streams[2]);
  ncclRecv(d_tempbufs[2], chunkSize * 4, ncclFloat, 3, comms[2], streams[2]);
  ncclGroupEnd();
    
  ncclGroupStart();
  ncclSend(d_buffers[3], chunkSize * 4, ncclFloat, 2, comms[3], streams[3]);
  ncclRecv(d_tempbufs[3], chunkSize * 4, ncclFloat, 2, comms[3], streams[3]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[4] + (4 * chunkSize), chunkSize * 4, ncclFloat, 5, comms[4], streams[4]);
  ncclRecv(d_tempbufs[4], chunkSize * 4, ncclFloat, 5, comms[4], streams[4]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[5], chunkSize * 4, ncclFloat, 4, comms[5], streams[5]);
  ncclRecv(d_tempbufs[5], chunkSize * 4, ncclFloat, 4, comms[5], streams[5]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[6] + (4 * chunkSize), chunkSize * 4, ncclFloat, 7, comms[6], streams[6]);
  ncclRecv(d_tempbufs[6], chunkSize * 4, ncclFloat, 7, comms[6], streams[6]);
  ncclGroupEnd();
    
  ncclGroupStart();
  ncclSend(d_buffers[7], chunkSize * 4, ncclFloat, 6, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize * 4, ncclFloat, 6, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupEnd();
  
  for (int r = 0; r < numRanks; ++r) {
    cudaSetDevice(devs[r]);
    int recvChunk = r % 2;
    float* dstPtr = d_buffers[r] + (4 * recvChunk * chunkSize);
    int numBlocks = (4 * chunkSize + RED_ADD_THREADS - 1) / RED_ADD_THREADS;
    reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[r]>>>(dstPtr, d_tempbufs[r], 4 * chunkSize);
  }

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0] + (2 * chunkSize), 2 * chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_tempbufs[0], 2 * chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[1] + (6 * chunkSize), 2 * chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclRecv(d_tempbufs[1], 2 * chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[2], 2 * chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclRecv(d_tempbufs[2], 2 * chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclGroupEnd();
    
  ncclGroupStart();
  ncclSend(d_buffers[3] + (4 * chunkSize), 2 * chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_tempbufs[3], 2 * chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclGroupEnd();

  ncclGroupStart(); 
  ncclSend(d_buffers[4] + (2 * chunkSize), 2 * chunkSize, ncclFloat, 6, comms[4], streams[4]);
  ncclRecv(d_tempbufs[4], 2 * chunkSize, ncclFloat, 6, comms[4], streams[4]);
  ncclGroupEnd();
  
  ncclGroupStart();
  ncclSend(d_buffers[5] + (6 * chunkSize), 2 * chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclRecv(d_tempbufs[5], 2 * chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[6], 2 * chunkSize, ncclFloat, 4, comms[6], streams[6]);
  ncclRecv(d_tempbufs[6], 2 * chunkSize, ncclFloat, 4, comms[6], streams[6]);
  ncclGroupEnd();
    
  ncclGroupStart();
  ncclSend(d_buffers[7] + (4 * chunkSize), 2 * chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], 2 * chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupEnd();

  int numBlocks = (2 * chunkSize + RED_ADD_THREADS - 1) / RED_ADD_THREADS;

  cudaSetDevice(devs[0]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[0]>>>(d_buffers[0], d_tempbufs[0], 2 * chunkSize);
  cudaSetDevice(devs[2]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[2]>>>(d_buffers[2] + (2 * chunkSize), d_tempbufs[2], 2 * chunkSize);
  cudaSetDevice(devs[1]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[1]>>>(d_buffers[1] + (4 * chunkSize), d_tempbufs[1], 2 * chunkSize);
  cudaSetDevice(devs[3]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[3]>>>(d_buffers[3] + (6 * chunkSize), d_tempbufs[3], 2 * chunkSize);

  cudaSetDevice(devs[4]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[4]>>>(d_buffers[4], d_tempbufs[4], 2 * chunkSize);
  cudaSetDevice(devs[6]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[6]>>>(d_buffers[6] + (2 * chunkSize), d_tempbufs[6], 2 * chunkSize);
  cudaSetDevice(devs[5]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[5]>>>(d_buffers[5] + (4 * chunkSize), d_tempbufs[5], 2 * chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + (6 * chunkSize), d_tempbufs[7], 2 * chunkSize);

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0] + chunkSize, chunkSize, ncclFloat, 4, comms[0], streams[0]);
  ncclRecv(d_tempbufs[0], chunkSize, ncclFloat, 4, comms[0], streams[0]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[1] + (5 * chunkSize), chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclRecv(d_tempbufs[1], chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[2] + (3 * chunkSize), chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclRecv(d_tempbufs[2], chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[3] + (7 * chunkSize), chunkSize, ncclFloat, 7, comms[3], streams[3]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 7, comms[3], streams[3]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[4], chunkSize, ncclFloat, 0, comms[4], streams[4]);
  ncclRecv(d_tempbufs[4], chunkSize, ncclFloat, 0, comms[4], streams[4]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[5] + (4 * chunkSize), chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclRecv(d_tempbufs[5], chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[6] + (2 * chunkSize), chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclRecv(d_tempbufs[6], chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[7] + (6 * chunkSize), chunkSize, ncclFloat, 3, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 3, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupEnd();

  numBlocks = (chunkSize + RED_ADD_THREADS - 1) / RED_ADD_THREADS;
  cudaSetDevice(devs[0]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[0]>>>(d_buffers[0], d_tempbufs[0], chunkSize);
  cudaSetDevice(devs[4]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[4]>>>(d_buffers[4] + chunkSize, d_tempbufs[4], chunkSize);

  cudaSetDevice(devs[2]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[2]>>>(d_buffers[2] + (2 * chunkSize), d_tempbufs[2], chunkSize);
  cudaSetDevice(devs[6]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[6]>>>(d_buffers[6] + (3 * chunkSize), d_tempbufs[6], chunkSize);

  cudaSetDevice(devs[1]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[1]>>>(d_buffers[1] + (4 * chunkSize), d_tempbufs[1], chunkSize);
  cudaSetDevice(devs[5]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[5]>>>(d_buffers[5] + (5 * chunkSize), d_tempbufs[5], chunkSize);

  cudaSetDevice(devs[3]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[3]>>>(d_buffers[3] + (6 * chunkSize), d_tempbufs[3], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + (7 * chunkSize), d_tempbufs[7], chunkSize);

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 4, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + chunkSize, chunkSize, ncclFloat, 4, comms[0], streams[0]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[1] + (4 * chunkSize), chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + (5 * chunkSize), chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[2] + (2 * chunkSize), chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclRecv(d_buffers[2] + (3 * chunkSize), chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[3] + (6 * chunkSize), chunkSize, ncclFloat, 7, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + (7 * chunkSize), chunkSize, ncclFloat, 7, comms[3], streams[3]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[4] + chunkSize, chunkSize, ncclFloat, 0, comms[4], streams[4]);
  ncclRecv(d_buffers[4], chunkSize, ncclFloat, 0, comms[4], streams[4]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[5] + (5 * chunkSize), chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + (4 * chunkSize), chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[6] + (3 * chunkSize), chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + (2 * chunkSize), chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[7] + (7 * chunkSize), chunkSize, ncclFloat, 3, comms[7], streams[7]);
  ncclRecv(d_buffers[7] + (6 * chunkSize), chunkSize, ncclFloat, 3, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupEnd();

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0], 2 * chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + (2 * chunkSize), 2 * chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[1] + (4 * chunkSize), 2 * chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + (6 * chunkSize), 2 * chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[2]  + (2 * chunkSize), 2 * chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclRecv(d_buffers[2], 2 * chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclGroupEnd();
    
  ncclGroupStart();
  ncclSend(d_buffers[3] + (6 * chunkSize), 2 * chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + (4 * chunkSize), 2 * chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[4], 2 * chunkSize, ncclFloat, 6, comms[4], streams[4]);
  ncclRecv(d_buffers[4] + (2 * chunkSize), 2 * chunkSize, ncclFloat, 6, comms[4], streams[4]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[5] + (4 * chunkSize), 2 * chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + (6 * chunkSize), 2 * chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[6] + (2 * chunkSize), 2 * chunkSize, ncclFloat, 4, comms[6], streams[6]);
  ncclRecv(d_buffers[6], 2 * chunkSize, ncclFloat, 4, comms[6], streams[6]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[7] + (6 * chunkSize), 2 * chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclRecv(d_buffers[7] + (4 * chunkSize), 2 * chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupEnd();

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize * 4, ncclFloat, 1, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + (4 * chunkSize), chunkSize * 4, ncclFloat, 1, comms[0], streams[0]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[1] + (4 * chunkSize), chunkSize * 4, ncclFloat, 0, comms[1], streams[1]);
  ncclRecv(d_buffers[1], chunkSize * 4, ncclFloat, 0, comms[1], streams[1]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[2], chunkSize * 4, ncclFloat, 3, comms[2], streams[2]);
  ncclRecv(d_buffers[2] + (4 * chunkSize), chunkSize * 4, ncclFloat, 3, comms[2], streams[2]);
  ncclGroupEnd();
  
  ncclGroupStart();
  ncclSend(d_buffers[3] + (4 * chunkSize), chunkSize * 4, ncclFloat, 2, comms[3], streams[3]);
  ncclRecv(d_buffers[3], chunkSize * 4, ncclFloat, 2, comms[3], streams[3]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[4], chunkSize * 4, ncclFloat, 5, comms[4], streams[4]);
  ncclRecv(d_buffers[4] + (4 * chunkSize), chunkSize * 4, ncclFloat, 5, comms[4], streams[4]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[5] + (4 * chunkSize), chunkSize * 4, ncclFloat, 4, comms[5], streams[5]);
  ncclRecv(d_buffers[5], chunkSize * 4, ncclFloat, 4, comms[5], streams[5]);
  ncclGroupEnd();

  ncclGroupStart();
  ncclSend(d_buffers[6], chunkSize * 4, ncclFloat, 7, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + (4 * chunkSize), chunkSize * 4, ncclFloat, 7, comms[6], streams[6]);
  ncclGroupEnd();
    
  ncclGroupStart();
  ncclSend(d_buffers[7] + (4 * chunkSize), chunkSize * 4, ncclFloat, 6, comms[7], streams[7]);
  ncclRecv(d_buffers[7], chunkSize * 4, ncclFloat, 6, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupEnd();

  cudaSetDevice(devs[0]);
  cudaEventRecord(stop, streams[0]);
  cudaEventSynchronize(stop);
}


void rhd_allreduce_delay(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size, clock_t sleep_cycles) {
  int chunkSize = size / numRanks;
  cudaSetDevice(devs[0]);
  cudaEventRecord(start, streams[0]);

  // sleep the straggler
  cudaSetDevice(devs[7]);
  gpu_sleep_kernel<<<1, 1, 0, streams[7]>>>(sleep_cycles);

  rhd_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}

void rhd_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size) {
  int chunkSize = size / numRanks;
  cudaSetDevice(devs[0]);
  cudaEventRecord(start, streams[0]);
  rhd_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}


void stragglar_allreduce_helper(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t chunkSize) {
  int numBlocks = (chunkSize + RED_ADD_THREADS - 1) / RED_ADD_THREADS;

  // Synchronize to make sure everything is idle
  for (int r = 0; r < numRanks; ++r) {
    cudaSetDevice(devs[r]);
    cudaStreamSynchronize(streams[r]);
  }

  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 7, comms[0], streams[0]);
  ncclRecv(d_tempbufs[0], chunkSize, ncclFloat, 7, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[7], chunkSize, ncclFloat, 0, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 0, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupEnd();
  cudaSetDevice(devs[0]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[0]>>>(d_buffers[0], d_tempbufs[0], chunkSize);
  
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7], d_tempbufs[7], chunkSize);

  // step 2
  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[1] + chunkSize, chunkSize, ncclFloat, 7, comms[1], streams[1]);
  ncclRecv(d_tempbufs[1], chunkSize, ncclFloat, 7, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[7] + chunkSize, chunkSize, ncclFloat, 1, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 1, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[3], chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupEnd();
 
  cudaSetDevice(devs[1]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[1]>>>(d_buffers[1] + chunkSize, d_tempbufs[1], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>(d_buffers[7] + chunkSize, d_tempbufs[7], chunkSize);

  // step 3
  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[7] + 2 * chunkSize, chunkSize, ncclFloat, 2, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 2, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[2] + 2 * chunkSize, chunkSize, ncclFloat, 7, comms[2], streams[2]);
  ncclRecv(d_tempbufs[2], chunkSize, ncclFloat, 7, comms[2], streams[2]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[1] + chunkSize, chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[4] + chunkSize, chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 5, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[5], chunkSize, ncclFloat, 0, comms[5], streams[5]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[3], chunkSize, ncclFloat, 6, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[6], chunkSize, ncclFloat, 3, comms[6], streams[6]);
  ncclGroupEnd();
  ncclGroupEnd();
  
  cudaSetDevice(devs[2]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[2]>>>(d_buffers[2] + 2 * chunkSize, d_tempbufs[2], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>( d_buffers[7] + 2 * chunkSize, d_tempbufs[7], chunkSize);

  // step 4
  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[7] + 3 * chunkSize, chunkSize, ncclFloat, 3, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 3, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[3] + 3 * chunkSize, chunkSize, ncclFloat, 7, comms[3], streams[3]);
  ncclRecv(d_tempbufs[3], chunkSize, ncclFloat, 7, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[0], chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + 2 * chunkSize, chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[2] + (2 * chunkSize), chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclRecv(d_buffers[2], chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[1] + chunkSize, chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclRecv(d_buffers[1], chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[5], chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + chunkSize, chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[4] + chunkSize, chunkSize, ncclFloat, 6, comms[4], streams[4]);
  ncclRecv(d_buffers[4], chunkSize, ncclFloat, 6, comms[4], streams[4]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[6], chunkSize, ncclFloat, 4, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + chunkSize, chunkSize, ncclFloat, 4, comms[6], streams[6]);
  ncclGroupEnd();
  ncclGroupEnd();
  cudaSetDevice(devs[3]);
  cudaStreamSynchronize(streams[3]);
  cudaStreamSynchronize(streams[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[3]>>>(d_buffers[3] + 3 * chunkSize, d_tempbufs[3], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>( d_buffers[7] + 3 * chunkSize, d_tempbufs[7], chunkSize);

  // step 5
  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[7] + 4 * chunkSize, chunkSize, ncclFloat, 4, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 4, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[4] + 4 * chunkSize, chunkSize, ncclFloat, 7, comms[4], streams[4]);
  ncclRecv(d_tempbufs[4], chunkSize, ncclFloat, 7, comms[4], streams[4]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[3] + 3 * chunkSize, chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + chunkSize, chunkSize, ncclFloat, 1, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[1] + chunkSize, chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + 3 * chunkSize, chunkSize, ncclFloat, 3, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[0] + 2 * chunkSize, chunkSize, ncclFloat, 5, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + chunkSize, chunkSize, ncclFloat, 5, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[5] + chunkSize, chunkSize, ncclFloat, 0, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + 2 * chunkSize, chunkSize, ncclFloat, 0, comms[5], streams[5]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[2] + 2 * chunkSize, chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclRecv(d_buffers[2] + chunkSize, chunkSize, ncclFloat, 6, comms[2], streams[2]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[6] + chunkSize, chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + 2 * chunkSize, chunkSize, ncclFloat, 2, comms[6], streams[6]);
  ncclGroupEnd();
  ncclGroupEnd();
  cudaSetDevice(devs[4]);
  cudaStreamSynchronize(streams[4]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[4]>>>(d_buffers[4] + 4 * chunkSize, d_tempbufs[4], chunkSize);
  cudaSetDevice(devs[7]);
  cudaStreamSynchronize(streams[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>( d_buffers[7] + 4 * chunkSize, d_tempbufs[7], chunkSize);

  // step 6
  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[7] + 5 * chunkSize, chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_tempbufs[5], chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclSend(d_buffers[5] + 5 * chunkSize, chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[4] + 4 * chunkSize, chunkSize, ncclFloat, 2, comms[4], streams[4]);
  ncclRecv(d_buffers[4] + 2 * chunkSize, chunkSize, ncclFloat, 2, comms[4], streams[4]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[2] + 4 * chunkSize, chunkSize, ncclFloat, 4, comms[2], streams[2]);
  ncclSend(d_buffers[2] + 2 * chunkSize, chunkSize, ncclFloat, 4, comms[2], streams[2]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[0] + 2 * chunkSize, chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + 3 * chunkSize, chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[3] + 2 * chunkSize, chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclSend(d_buffers[3] + 3 * chunkSize, chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[1] + 3 * chunkSize, chunkSize, ncclFloat, 6, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + 2 * chunkSize, chunkSize, ncclFloat, 6, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[6] + 3 * chunkSize, chunkSize, ncclFloat, 1, comms[6], streams[6]);
  ncclSend(d_buffers[6] + 2 * chunkSize, chunkSize, ncclFloat, 1, comms[6], streams[6]);
  ncclGroupEnd();
  ncclGroupEnd();

  cudaSetDevice(devs[5]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[5]>>>(d_buffers[5] + 5 * chunkSize, d_tempbufs[5], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>( d_buffers[7] + 5 * chunkSize, d_tempbufs[7], chunkSize);

  // step 7
  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[7] + 6 * chunkSize, chunkSize, ncclFloat, 6, comms[7], streams[7]);
  ncclRecv(d_tempbufs[7], chunkSize, ncclFloat, 6, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_tempbufs[6], chunkSize, ncclFloat, 7, comms[6], streams[6]);
  ncclSend(d_buffers[6] + 6 * chunkSize, chunkSize, ncclFloat, 7, comms[6], streams[6]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[5] + 5 * chunkSize, chunkSize, ncclFloat, 3, comms[5], streams[5]);
  ncclRecv(d_buffers[5] + 3 * chunkSize, chunkSize, ncclFloat, 3, comms[5], streams[5]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[3] + 5 * chunkSize, chunkSize, ncclFloat, 5, comms[3], streams[3]);
  ncclSend(d_buffers[3] + 3 * chunkSize, chunkSize, ncclFloat, 5, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[0] + 3 * chunkSize, chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + 4 * chunkSize, chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[2] + 3 * chunkSize, chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclSend(d_buffers[2] + 4 * chunkSize, chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[1] + 3 * chunkSize, chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + 4 * chunkSize, chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[4] + 3 * chunkSize, chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclSend(d_buffers[4] + 4 * chunkSize, chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclGroupEnd();
  ncclGroupEnd();
  cudaSetDevice(devs[6]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[6]>>>(d_buffers[6] + 6 * chunkSize, d_tempbufs[6], chunkSize);
  cudaSetDevice(devs[7]);
  reduce_add<<<numBlocks, RED_ADD_THREADS, 0, streams[7]>>>( d_buffers[7] + 6 * chunkSize, d_tempbufs[7], chunkSize);

  // step 8
  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[6] + 6 * chunkSize, chunkSize, ncclFloat, 4, comms[6], streams[6]);
  ncclRecv(d_buffers[6] + 4 * chunkSize, chunkSize, ncclFloat, 4, comms[6], streams[6]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[4] + 6 * chunkSize, chunkSize, ncclFloat, 6, comms[4], streams[4]);
  ncclSend(d_buffers[4] + 4 * chunkSize, chunkSize, ncclFloat, 6, comms[4], streams[4]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[0] + 4 * chunkSize, chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + 5 * chunkSize, chunkSize, ncclFloat, 3, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[3] + 4 * chunkSize, chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclSend(d_buffers[3] + 5 * chunkSize, chunkSize, ncclFloat, 0, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[1] + 4 * chunkSize, chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + 5 * chunkSize, chunkSize, ncclFloat, 5, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[5] + 4 * chunkSize, chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclSend(d_buffers[5] + 5 * chunkSize, chunkSize, ncclFloat, 1, comms[5], streams[5]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[7] + 6 * chunkSize, chunkSize, ncclFloat, 2, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[2] + 6 * chunkSize, chunkSize, ncclFloat, 7, comms[2], streams[2]);
  ncclGroupEnd();
  ncclGroupEnd();

  // step 9
  ncclGroupStart();
  ncclGroupStart();
  ncclSend(d_buffers[0] + 5 * chunkSize, chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclRecv(d_buffers[0] + 6 * chunkSize, chunkSize, ncclFloat, 2, comms[0], streams[0]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[2] + 5 * chunkSize, chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclSend(d_buffers[2] + 6 * chunkSize, chunkSize, ncclFloat, 0, comms[2], streams[2]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[1] + 5 * chunkSize, chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclRecv(d_buffers[1] + 6 * chunkSize, chunkSize, ncclFloat, 4, comms[1], streams[1]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[4] + 5 * chunkSize, chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclSend(d_buffers[4] + 6 * chunkSize, chunkSize, ncclFloat, 1, comms[4], streams[4]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[3] + 5 * chunkSize, chunkSize, ncclFloat, 6, comms[3], streams[3]);
  ncclRecv(d_buffers[3] + 6 * chunkSize, chunkSize, ncclFloat, 6, comms[3], streams[3]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[6] + 5 * chunkSize, chunkSize, ncclFloat, 3, comms[6], streams[6]);
  ncclSend(d_buffers[6] + 6 * chunkSize, chunkSize, ncclFloat, 3, comms[6], streams[6]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclSend(d_buffers[7] + 6 * chunkSize, chunkSize, ncclFloat, 5, comms[7], streams[7]);
  ncclGroupEnd();
  ncclGroupStart();
  ncclRecv(d_buffers[5] + 6 * chunkSize, chunkSize, ncclFloat, 7, comms[5], streams[5]);
  ncclGroupEnd();
  ncclGroupEnd();

  cudaSetDevice(devs[0]);
  cudaEventRecord(stop, streams[0]);
  cudaEventSynchronize(stop);
}

void stragglar_allreduce_delay(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, ncclComm_t* subComms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size, clock_t sleep_cycles) {
  size_t chunkSize = size / (numRanks - 1);
  cudaSetDevice(devs[0]);
  cudaEventRecord(start, streams[0]);

  cudaSetDevice(devs[7]);
  gpu_sleep_kernel<<<1, 1, 0, streams[7]>>>(sleep_cycles);

  // Synchronize to make sure everything is idle
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

  stragglar_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}

void stragglar_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, cudaEvent_t start, cudaEvent_t stop, int numRanks, size_t size) {
    size_t chunkSize = size / (numRanks - 1);
    cudaSetDevice(devs[0]);
    cudaEventRecord(start, streams[0]);
    stragglar_allreduce_helper(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, chunkSize);
}

int main(int argc, char* argv[]) {
  const int numRanks = 8;

  int version;
  ncclGetVersion(&version);

  // printf("NCCL version %d\n", version);
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <bufferSize> <algorithm> <numIters> <sleepTimeMs>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  size_t bytes = (size_t)strtoull(argv[1], NULL, 10);
  size_t size = bytes / sizeof(float);
  const char* alg = argv[2];
  int numIters = atoi(argv[3]);
  float sleepTime = atof(argv[4]);

  if (strcmp(alg, "ring") != 0 && strcmp(alg, "rhd") != 0 &&
      strcmp(alg, "direct") != 0 && strcmp(alg, "stragglar") != 0) {
    fprintf(stderr, "Invalid algorithm: %s\n", alg);
    exit(EXIT_FAILURE);
  }

  // Check GPUs
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
  ncclComm_t comms[numRanks];
  ncclComm_t subComms[numRanks - 1];
 
  clock_t sleep_cycles;
  if (sleepTime >= 0) {
    sleep_cycles = calculate_sleep_cycles(sleepTime, devs);
    printf("Sleep cycles: %ld\n", sleep_cycles);
  }

  cudaSetDevice(devs[0]);
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_NCCL(ncclCommInitAll(comms, numRanks, devs));
  if (sleepTime >= 0) {
    CHECK_NCCL(ncclCommInitAll(subComms, numRanks - 1, NULL));
  }

  size_t chunkSize;

  if (strcmp(argv[2], "stragglar") == 0) {
    chunkSize = size / (numRanks - 1);
  }
  else if (strcmp(argv[2], "direct") == 0) {
    chunkSize = size;
  }
  else {
    chunkSize = size / numRanks;
  }

  for (int i = 0; i < numRanks; ++i) {
    CHECK_CUDA(cudaSetDevice(devs[i]));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaMallocAsync(&d_buffers[i], size * sizeof(float), streams[i]));
    if (strcmp(argv[2], "ring") == 0 || strcmp(argv[2], "stragglar") == 0) {
      CHECK_CUDA(cudaMallocAsync(&d_tempbufs[i], chunkSize * sizeof(float), streams[i]));
    }
    else if (strcmp(argv[2], "rhd") == 0) {
      CHECK_CUDA(cudaMallocAsync(&d_tempbufs[i], 4 * chunkSize * sizeof(float), streams[i]));
    }
    else if (strcmp(argv[2], "direct") == 0) {
      CHECK_CUDA(cudaMallocAsync(&d_tempbufs[i], size * sizeof(float), streams[i]));
    }
  }

  // warmup
  std::vector<std::vector<std::pair<int, int>>> steps = {{ {0,3}, {2,6}, {1,5}, {4,7} }};
  for (int iter = 0; iter < 10; ++iter) {
    for (const auto& step : steps) {
      ncclGroupStart();
      for (const auto& [src, dst] : step) {
        ncclSend(d_buffers[src], chunkSize, ncclFloat, dst, comms[src], streams[src]);
        ncclRecv(d_tempbufs[src], chunkSize, ncclFloat, dst, comms[src], streams[src]);
  
        ncclSend(d_buffers[dst], chunkSize, ncclFloat, src, comms[dst], streams[dst]);
        ncclRecv(d_tempbufs[dst], chunkSize, ncclFloat, src, comms[dst], streams[dst]);
      }
      ncclGroupEnd();
    }
  }

  printf("algorithm,buffer_size_bytes,iteration,delay,runtime_ms,BW(GB/s)\n");
  for (int iter = 0; iter < numIters + 1; ++iter) {
    // Reset buffers if needed (same init pattern as above)
    for (int i = 0; i < numRanks; ++i) {
      CHECK_CUDA(cudaSetDevice(devs[i]));
      if (sleepTime < 0 && strcmp(alg,"direct")==0 && i < numRanks-1) {
        fill_pattern<<<(size+255)/256, 256, 0, streams[i] >>>(d_buffers[i], 28.f, size);
      } else{
        fill_pattern<<<(size+255)/256, 256, 0, streams[i] >>>(d_buffers[i], float(i+1), size);
      }
      if (sleepTime < 0 && strcmp(alg, "stragglar") == 0 && i < numRanks - 1) {
        fill_pattern<<< (chunkSize+255)/256, 256, 0, streams[i] >>>(d_buffers[i] + i*chunkSize, 28.f, chunkSize);
      }
    }

    for (int i = 0; i < numRanks; ++i) {
      CHECK_CUDA(cudaSetDevice(devs[i]));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

  // Run algorithm
    if (sleepTime >= 0) {
      if (strcmp(alg, "ring") == 0)
        ring_allreduce_delay(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size, sleep_cycles);
      else if (strcmp(alg, "rhd") == 0)
        rhd_allreduce_delay(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size, sleep_cycles);
      else if (strcmp(alg, "stragglar") == 0)
        stragglar_allreduce_delay(d_buffers, d_tempbufs, devs, streams, comms, subComms, start, stop, numRanks, size, sleep_cycles);
      else if (strcmp(alg, "direct") == 0)
        direct_allreduce_delay(d_buffers, d_tempbufs, devs, streams, comms, subComms, start, stop, numRanks, size, sleep_cycles);
    } else {
      if (strcmp(alg, "ring") == 0)
        ring_allreduce(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size);
      else if (strcmp(alg, "rhd") == 0) 
        rhd_allreduce(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size);
      else if (strcmp(alg, "stragglar") == 0) {
        stragglar_allreduce(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size);
      }
      else if (strcmp(alg, "direct") == 0)
        direct_allreduce(d_buffers, d_tempbufs, devs, streams, comms, start, stop, numRanks, size);
    }
    float ms;
    float bw;
    cudaSetDevice(devs[0]);
    cudaEventElapsedTime(&ms, start, stop);
    if (iter == 0) continue;
    if (sleepTime > 0) {
      bw = (float)size * sizeof(float) / 1024.0 / 1024.0 / 1024.0 * 1000.0 / (ms - sleepTime);
    }
    else {
      bw = (float)size * sizeof(float) / 1024.0 / 1024.0 / 1024.0 * 1000.0 / ms;
    }
    printf("%s,%zu,%d,%.3f,%.3f,%.3f\n",
      alg,
      (size_t)size * sizeof(float),   // bytes, still a size_t
      iter,
      sleepTime,
      ms,
      bw);
  }
  
  float* hostOut = (float*)malloc(size * sizeof(float));
  for (int r = 0; r < numRanks; ++r) {
    CHECK_CUDA(cudaSetDevice(devs[r]));
    CHECK_CUDA(cudaMemcpy(hostOut, d_buffers[r],
                          size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
      assert(hostOut[i] == 36.0);
    }
  }
  free(hostOut);

  for (int i = 0; i < numRanks; ++i) {
    cudaSetDevice(devs[i]);
    cudaFree(d_buffers[i]);
    cudaFree(d_tempbufs[i]);
    cudaStreamDestroy(streams[i]);  // Streams are local to each device
    ncclCommDestroy(comms[i]);      // Safe last
    if (sleepTime >= 0 && i < numRanks - 1) {
      ncclCommDestroy(subComms[i]);
    }
    printf("Rank %d, done\n", i);
  }
  

  return 0;
}