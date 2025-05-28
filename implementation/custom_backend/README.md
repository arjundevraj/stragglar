## Build
```python setup.py develop```

## Testing
```
NCCL_DEBUG=TRACE torchrun --nproc-per-node=<#nodes> example.py
```

## Usage
```
torchrun --nproc-per-node=<#nodes> dp.py
```