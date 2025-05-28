#include "dummy.hpp"
#include <iostream>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <torch/torch.h> 
#include <torch/extension.h>
#include <cuda.h>

namespace net {

inline ssize_t send(int sockfd, const void* buf, size_t len, int flags) {
    return ::send(sockfd, buf, len, flags);
}

inline ssize_t recv(int sockfd, void* buf, size_t len, int flags) {
    return ::recv(sockfd, buf, len, flags);
}

} 

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

extern void launch_reduce_add(float* dst, const float* src, int n,
  cudaStream_t stream);

static uint64_t getHostHash(const char* string) {
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

void setSocketReuse(int sockfd) {
    int opt = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("Error setting SO_REUSEADDR");
        exit(EXIT_FAILURE);
    }
}

// Function to establish a server socket (TCP)
int createServerSocket(int port) {
    int sockfd;
    struct sockaddr_in server_addr;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Error creating socket");
        exit(EXIT_FAILURE);
    }

    setSocketReuse(sockfd);
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Error binding socket");
        exit(EXIT_FAILURE);
    }

    if (listen(sockfd, 100) < 0) {
        perror("Error listening on socket");
        exit(EXIT_FAILURE);
    }

    return sockfd;
}

// Function to establish a client socket (TCP)
int createClientSocket(const char *server_ip, int port) {
    int sockfd;
    struct sockaddr_in server_addr;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Error creating socket");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        perror("Invalid address or address not supported");
        exit(EXIT_FAILURE);
    }

    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        exit(EXIT_FAILURE);
    }

    return sockfd;
}

namespace c10d {


bool WorkDummy::isCompleted() {
  return true;
}

bool WorkDummy::isSuccess() const {
  return true;
}

bool WorkDummy::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkDummy::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
BackendDummy::BackendDummy(int rank, int size)
    : Backend(rank, size) {
    int STRAGGLER_RANK = 6;
    size_ = size;
    rank_ = rank;

    int device_id = rank;
    if (rank == STRAGGLER_RANK) {
      device_id = size - 1;
    } else if (rank == size - 1) {
      device_id = STRAGGLER_RANK;
    }

    cudaError_t err = cudaSetDevice(device_id);

    const char* server_ip = "0.0.0.0";

    int clientSockets[size_-1];
    int sockfd;
    int port = 34567; // Port for communication

    
    if (rank_ == 0) {
      // Rank 0 acts as the server
      sockfd = createServerSocket(port);

      printf("Rank 0 waiting for %d connections...\n", size_ - 1);

      for (int i = 0; i < size_ - 1; i++) {
        clientSockets[i] = accept(sockfd, NULL, NULL);
        if (clientSockets[i] < 0) {
            perror("Error accepting connection");
            exit(EXIT_FAILURE);
        }
        printf("Accepted connection %d\n", i + 1);
      }
    } else {
      sleep(1);
      printf("Connecting to %s:%d\n", server_ip, port);
      sockfd = createClientSocket(server_ip, port);  // Change IP to server's IP
      printf("Connected to %s:%d\n", server_ip, port);
    }

    ncclUniqueId id;
    if (rank_ == 0) {
      ncclGetUniqueId(&id);

      for (int i = 0; i < size_ - 1; i++) {
        net::send(clientSockets[i], &id, sizeof(id), 0);
        close(clientSockets[i]);
      }

      close(sockfd);
    } else {
      net::recv(sockfd, &id, sizeof(id), 0);
      close(sockfd);
    }
    // assume id has been broadcast to all ranks
    NCCLCHECK(ncclCommInitRank(&comm_, size_, id, rank_));
    NCCLCHECK(ncclCommSplit(comm_, rank_<size_-1 ? 0 : NCCL_SPLIT_NOCOLOR, rank_, &subcomm_, NULL));

    //initialize stream
    streams_.resize(size_);
    for (int i = 0; i < size_; ++i) {
      CUDACHECK(cudaStreamCreate(&streams_[i]));
    }
}

BackendDummy::~BackendDummy() {
  ncclCommDestroy(comm_);
  ncclCommDestroy(subcomm_);
  for (int i = 0; i < size_; ++i) {
    CUDACHECK(cudaStreamDestroy(streams_[i]));
  }
}

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  for (auto& outputTensorVec : outputTensors) {
      for (auto& outputTensor : outputTensorVec) {
          outputTensor.zero_();
      }
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

void allreduce_stragglar(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts,
  int rank,
  int size,
  ncclComm_t subComm,
  ncclComm_t comm,
  cudaStream_t stream
) {
  auto recvtensor = at::empty(tensors[0].sizes(), tensors[0].options());
  int chunkSize = tensors[0].numel()/(size - 1);
  if (rank != size - 1) {
    ncclGroupStart();
    ncclReduceScatter((float *) tensors[0].data_ptr(), (float *) tensors[0].data_ptr() + (rank * chunkSize), chunkSize, ncclFloat, ncclSum, subComm, stream);
    ncclGroupEnd();
  }

  if (rank == 0) {
    // step 1 
    ncclGroupStart();
    ncclSend(tensors[0].data_ptr(), chunkSize, ncclFloat, 7, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>(), recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 2
    ncclGroupStart();
    ncclSend(tensors[0].data_ptr(), chunkSize, ncclFloat, 3, comm, stream);
    ncclGroupEnd();

    // step 3
    ncclGroupStart();
    ncclSend(tensors[0].data_ptr(), chunkSize, ncclFloat, 5, comm, stream);
    ncclGroupEnd();

    // step 4
    ncclGroupStart();
    ncclSend(tensors[0].data_ptr(), chunkSize, ncclFloat, 2, comm, stream);
    ncclRecv((float *)tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclGroupEnd();

    // step 5
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclGroupEnd();

    // step 6
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclGroupEnd();

    // step 7
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr()+ 4 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclGroupEnd();

    // step 8
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclGroupEnd();

    // step 9
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclGroupEnd();
  } 
  else if (rank == 1) {
    // step 2
    ncclGroupStart();
    ncclSend((float *)tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 7, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();

    launch_reduce_add(tensors[0].data_ptr<float>() + chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);
    
    // step 3
    ncclGroupStart();
    ncclSend((float *)tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclGroupEnd();
    
    // step 4
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr(), chunkSize, ncclFloat, 5, comm, stream);
    ncclGroupEnd();

    // step 5
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclGroupEnd();

    // step 6
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclGroupEnd();

    // step 7
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclGroupEnd();

    // step 8
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclGroupEnd();

    // step 9
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclGroupEnd();
  }
  else if (rank == 2) {
    
    // step 3
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 7, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + 2 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 4
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + (2 * chunkSize), chunkSize, ncclFloat, 0, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr(), chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();

    // step 5
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclRecv( (float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclGroupEnd();

    // step 6
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclGroupEnd();

    // step 7
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();

    // step 8
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();

    // step 9
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();
  }
  else if (rank == 3) {
    // step 2
    ncclGroupStart();
    ncclRecv(tensors[0].data_ptr(), chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();

    // step 3
    ncclGroupStart();
    ncclSend(tensors[0].data_ptr(), chunkSize, ncclFloat, 6, comm, stream);
    ncclGroupEnd();

    // step 4
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 7, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();

    launch_reduce_add(tensors[0].data_ptr<float>() + 3 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 5
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclGroupEnd();

    // step 6
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();

    // step 7
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclGroupEnd();
    
    // step 8
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();

    // step 9
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr()  + 5 * chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclGroupEnd();
  }
  else if (rank == 4) {
    // step 3
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclGroupEnd();

    // step 4
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr(), chunkSize, ncclFloat, 6, comm, stream);
    ncclGroupEnd();

    // step 5
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 7, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + 4 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 6
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclGroupEnd();

    // step 7
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclGroupEnd();
    
    // step 8
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclGroupEnd();

    // step 9
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclGroupEnd();

  }
  else if (rank == 5) {
    // step 3
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr(), chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();

    // step 4
    ncclGroupStart();
    ncclSend(tensors[0].data_ptr(), chunkSize, ncclFloat, 1, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclGroupEnd();

    // step 5
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();

    // step 6
    ncclGroupStart();
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 7, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + 5 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 7
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr()  + 5 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr()  + 3 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclGroupEnd();

    // step 8
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclGroupEnd();

    // step 9
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();
  }
  else if (rank == 6) {
    ncclGroupStart();
    ncclRecv(tensors[0].data_ptr(), chunkSize, ncclFloat, 3, comm, stream);
    ncclGroupEnd();

    // step 4
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr(), chunkSize, ncclFloat, 4, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclGroupEnd();
    
    // step 5
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclRecv((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclGroupEnd();

    // step 6
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclGroupEnd();

    // step 7
    ncclGroupStart();
    ncclRecv((float*) recvtensor.data_ptr(), chunkSize, ncclFloat, 7, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 7, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + 6 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 8
    ncclGroupStart();
    ncclSend((float*) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclRecv((float*) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclGroupEnd();

    // step 9
    ncclGroupStart();
    ncclRecv((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclSend((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclGroupEnd();
    
  }
  else if (rank == 7) {
    ncclGroupStart();
    ncclSend(tensors[0].data_ptr(), chunkSize, ncclFloat, 0, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 0, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>(), recvtensor.data_ptr<float>(), chunkSize, stream);
    //step 2
    ncclGroupStart();
    ncclSend((float *)tensors[0].data_ptr() + chunkSize, chunkSize, ncclFloat, 1, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 1, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);
    // step 3
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 2 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 2, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + 2 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 4
    ncclGroupStart();
    ncclSend((float *)tensors[0].data_ptr() + 3 * chunkSize, chunkSize, ncclFloat, 3, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 3, comm, stream);
    ncclGroupEnd();

    launch_reduce_add(tensors[0].data_ptr<float>() + 3 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);
    
    // step 5
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 4 * chunkSize, chunkSize, ncclFloat, 4, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 4, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + 4 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 6
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 5 * chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 5, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + 5 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 7
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 6, comm, stream);
    ncclRecv(recvtensor.data_ptr(), chunkSize, ncclFloat, 6, comm, stream);
    ncclGroupEnd();
    launch_reduce_add(tensors[0].data_ptr<float>() + 6 * chunkSize, recvtensor.data_ptr<float>(), chunkSize, stream);

    // step 8
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 2, comm, stream);
    ncclGroupEnd();

    // step 9
    ncclGroupStart();
    ncclSend((float *) tensors[0].data_ptr() + 6 * chunkSize, chunkSize, ncclFloat, 5, comm, stream);
    ncclGroupEnd();
  }

}

void allreduce_ring(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts,
  int rank,
  int size,
  ncclComm_t comm,
  cudaStream_t stream
) {
  auto recvtensor = at::empty(tensors[0].sizes(), tensors[0].options());
  int chunkSize = tensors[0].numel()/size;
    // Ring Reduce-Scatter
    for (int step = 1; step < size; ++step) {
      for (int r = 0; r < size; ++r) {
        if (r != rank) {
          continue;
        }
        int sendTo = (r + 1) % size;
        int recvFrom = (r - 1 + size) % size;
        int sendChunk = (r - step + size) % size;

        float* sendPtr = (float *) tensors[0].data_ptr() + sendChunk * chunkSize;
        float* recvPtr = (float *) recvtensor.data_ptr();
        ncclGroupStart();
        ncclSend(sendPtr, chunkSize, ncclFloat, sendTo, comm, stream);
        ncclRecv(recvPtr, chunkSize, ncclFloat, recvFrom, comm, stream);
        ncclGroupEnd();
      }
      for (int r = 0; r < size; ++r) {
        if (r != rank) {
          continue;
        }
        int recvChunk = (r - step - 1 + size) % size;
        launch_reduce_add((float *) tensors[0].data_ptr<float>() + (recvChunk * chunkSize), recvtensor.data_ptr<float>(), chunkSize, stream);
      }
    }
    for (int step = 0; step < size - 1; ++step) {
      for (int r = 0; r < size; ++r) {
        if (r != rank) {
          continue;
        }

        int sendTo = (r + 1) % size;
        int recvFrom = (r - 1 + size) % size;
        int sendChunk = (r - step + size) % size;
        int recvChunk = (r - step - 1 + size) % size;

        float* sendPtr = (float *) tensors[0].data_ptr() + sendChunk * chunkSize;
        float* recvPtr = (float *) tensors[0].data_ptr() + recvChunk * chunkSize;
        ncclGroupStart();
        ncclSend(sendPtr, chunkSize, ncclFloat, sendTo, comm, stream);
        ncclRecv(recvPtr, chunkSize, ncclFloat, recvFrom, comm, stream);
        ncclGroupEnd();
      }
    }
}


c10::intrusive_ptr<Work> BackendDummy::allreduce(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts) {

  // Toggle between StragglAR and Ring AllReduce depending on which you want to use

  allreduce_stragglar(tensors, opts, rank_, size_, subcomm_, comm_, streams_[rank_]);
  // allreduce_ring(tensors, opts, rank_, size_, comm_, streams_[rank_]);
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::barrier(
    const BarrierOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {

  
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<BackendDummy>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createBackendDummy", &BackendDummy::createBackendDummy);
}

} // namespace c10d
