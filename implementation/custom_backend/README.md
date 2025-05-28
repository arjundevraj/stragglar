# Implementation
StragglAR is implemented as a [custom `torch.distributed` backend](https://docs.pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html) in C++ and CUDA using the NCCL P2P API.
StragglAR is easy to integrate with existing `PyTorch` applications. Applications can simply bind the `torch.distributed` communication backend to StragglAR backend during `torch.distributed` environment initialization, as shown in [example.py](./example.py).

## Build  
```bash  
python setup.py develop  
```  

## Test  
```bash  
NCCL_DEBUG=TRACE torchrun --nproc-per-node=<#nodes> example.py  
```  

## Using StragglAR for LLM workloads  
```bash  
torchrun --nproc-per-node=<#nodes> dp.py  
```  