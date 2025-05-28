import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
from datasets import Dataset
import torch.cuda.nvtx as nvtx
from datasets import load_from_disk
from torch.cuda.amp import autocast, GradScaler
import dummy_collectives
import math

STRAGGLAR_RANK = 6
device_rank = 0

rank_colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
reset_color = "\033[0m"

def flatten_tensors(tensor_list):
    """Flatten a list of tensors into one contiguous 1D buffer."""
    buffer = torch.cat([t.contiguous().view(-1) for t in tensor_list])
    numel = buffer.numel()
    # resize_numel = math.ceil(numel * 4/(1024 * 7)) * 1024 * 7 // 4
    resize_numel = numel

    padded_buffer = torch.zeros(resize_numel, dtype=buffer.dtype, device=buffer.device)
    padded_buffer[:numel] = buffer
    
    return padded_buffer

def unflatten_tensors(flat, tensor_list):
    """Unflatten a contiguous 1D buffer into the original tensor shapes."""
    outputs = []
    offset = 0
    for tensor in tensor_list:
        numel = tensor.numel()
        outputs.append(flat[offset:offset+numel].view_as(tensor))
        offset += numel
    return outputs

def allreduce_gradients(model, world_size):
    grads = [p.grad.data for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads:
        return

    # Flatten all gradients
    flat_grads = flatten_tensors(grads)

    # All-reduce once
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads /= world_size

    # Unflatten and copy back to the individual gradients
    for p, g in zip([p for p in model.parameters() if p.requires_grad and p.grad is not None],
                    unflatten_tensors(flat_grads, grads)):
        p.grad.data.copy_(g)


def setup_ddp(rank, world_size):
    

    local_rank = int(os.environ["LOCAL_RANK"])
    device_rank = local_rank
    if local_rank == STRAGGLAR_RANK:
        device_rank = world_size - 1
    elif local_rank == world_size - 1:
        device_rank = STRAGGLAR_RANK
    dev_id = torch.device('cuda', device_rank)
    dist.init_process_group(backend="dummy", rank=rank, world_size=world_size, device_id=dev_id)
    torch.cuda.set_device(device_rank)
    color = rank_colors[rank % len(rank_colors)]
    
    print(f"{color}Rank {rank} | device_id: {dev_id}")
    return dev_id

def cleanup_ddp():
    dist.destroy_process_group()

def format_example(example, tokenizer):
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input"):
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

def train_ddp(rank, world_size):
    device = setup_ddp(rank, world_size)
    # print(f"rank {rank} | Current device: {device}")

    model_path = "./Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    # Load from local alpaca dataset
    dataset_path = "./alpaca"  # Path to local dataset directory
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path).select(range(8192))
    else:
        raise FileNotFoundError(f"Local Alpaca dataset not found at {dataset_path}")
    train_dataset = dataset
    # eval_dataset = dataset["test"]

    train_dataset = train_dataset.map(lambda ex: format_example(ex, tokenizer), remove_columns=train_dataset.column_names)
    # eval_dataset = eval_dataset.map(lambda ex: format_example(ex, tokenizer), remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, collate_fn=data_collator)
    # eval_loader = DataLoader(eval_da6taset, batch_size=16, shuffle=False, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # accelerator = Accelerator()

    # Set up optimizer
    # optimizer = FusedAdam(
    #     model.parameters(),
    #     lr=5e-5,
    #     betas=(0.9, 0.98),
    #     weight_decay=0.01,
    #     eps=1e-8
    # )

    # # Add optimizer to accelerator
    # optimizer = accelerator.prepare(optimizer)

    EPOCH = 1
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_loader) * EPOCH
    )

    itr = 0
    # lora_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     r=4096,
    #     lora_alpha=8,
    #     lora_dropout=0.05,
    #     target_modules=["q_proj", "v_proj"]
    # )
    # model = get_peft_model(model, lora_config)
    model.train()
    for epoch in range(EPOCH):
        # torch.cuda.empty_cache() 
        nvtx.range_push(f"Epoch {epoch + 1}")
        train_sampler.set_epoch(epoch)
        epoch_loss = 0

        for step, batch in enumerate(train_loader):
            nvtx.range_push(f"Step {step + 1}")
            iteration_start_time = torch.cuda.Event(enable_timing=True)
            iteration_end_time = torch.cuda.Event(enable_timing=True)
            forward_start_time = torch.cuda.Event(enable_timing=True)
            forward_end_time = torch.cuda.Event(enable_timing=True)
            backward_start_time = torch.cuda.Event(enable_timing=True)
            backward_end_time = torch.cuda.Event(enable_timing=True)
            ar_start_time = torch.cuda.Event(enable_timing=True)
            ar_end_time = torch.cuda.Event(enable_timing=True)

            iteration_start_time.record()
            forward_start_time.record()
            nvtx.range_push("Forward")
            outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["input_ids"].to(device))
            # with autocast():
            #     outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["input_ids"].to(device))
            #     loss = outputs.loss
            loss = outputs.loss


            torch.cuda.synchronize(device=device)
            nvtx.range_pop()
            forward_end_time.record()

            backward_start_time.record()
            nvtx.range_push("Backward")
            loss.backward()
            # scaler.scale(loss).backward()
            torch.cuda.synchronize(device=device)
            nvtx.range_pop()
            backward_end_time.record()

            nvtx.range_push("AllReduce_Gradients")
            total_buffer_size = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Record buffer size for communication
                    buffer_size = param.grad.data.numel() * param.grad.data.element_size()
                    total_buffer_size += buffer_size

            ar_start_time.record()
            allreduce_gradients(model, world_size)
                    
            ar_end_time.record()
            torch.cuda.synchronize(device=device)
            nvtx.range_pop()

            ar_time = ar_start_time.elapsed_time(ar_end_time)

            if rank == 0:
                print(f"Rank {rank} | AR Kernel Time: {ar_time:.2f} ms | AR Kernel Buffer Size: {total_buffer_size / 1024:.2f} KB | AR Kernel resize buffer size: {total_buffer_size / 1024:.2f} KB \n")

            # nvtx.range_push("AllReduce_Gradients")
            # total_buffer_size = 0
            # ar_start_time.record()

            
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         nvtx.range_push(f"allreduce:{name}")
            #         ar_kernel_start_time = torch.cuda.Event(enable_timing=True)
            #         ar_kernel_end_time = torch.cuda.Event(enable_timing=True)
                    
            #         numel = param.grad.data.numel()
            #         # resize_numel = math.ceil(numel * 4/(1024 * 7)) * 1024 * 7 // 4
            #         resize_numel = numel
                    
            #         # param.grad.data.resize_(resize_numel)
            #         grad = param.grad.data
            #         original_shape = grad.shape
            #         grad_flat = grad.view(-1)

            #         padded_grad = torch.zeros(resize_numel, dtype=grad.dtype, device=grad.device)
            #         padded_grad[:numel] = grad_flat

            #         ar_kernel_start_time.record()
            #         # dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            #         dist.all_reduce(padded_grad, op=dist.ReduceOp.SUM)

            #         torch.cuda.synchronize(device=device)
                    
            #         ar_kernel_end_time.record()

            #         # param.grad.data /= world_size
            #         padded_grad /= world_size
            #         nvtx.range_pop()
            #         # Record buffer size for communication
            #         # buffer_size = param.grad.data.numel() * param.grad.data.element_size()
            #         buffer_size = padded_grad.numel() * padded_grad.element_size()

            #         # param.grad.data.resize_(numel)
            #         # grad = padded_grad[:numel]
            #         param.grad.data.copy_(padded_grad[:numel].view(original_shape))

            #         if rank == 0:
            #             print(f"Rank {rank} | AR Kernel Time: {ar_kernel_start_time.elapsed_time(ar_kernel_end_time):.2f} ms | AR Kernel Buffer Size: {buffer_size / 1024:.2f} KB | AR Kernel resize buffer size: {resize_numel / 1024 * 4:.2f} KB \n")

            #         total_buffer_size += buffer_size
                    
            # torch.cuda.synchronize(device=device)
            # ar_end_time.record()
            # nvtx.range_pop()

            # scaler.step(optimizer)
            # scaler.update()
            # lr_scheduler.step()
            # optimizer.zero_grad()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            torch.cuda.synchronize(device=device)
            iteration_end_time.record()
            torch.cuda.synchronize(device=device)

            forward_time = forward_start_time.elapsed_time(forward_end_time)
            backward_time = backward_start_time.elapsed_time(backward_end_time)
            iteration_time = iteration_start_time.elapsed_time(iteration_end_time)
            # ar_time = ar_start_time.elapsed_time(ar_end_time)

            color = rank_colors[rank % len(rank_colors)]
            print(f"{color}Rank {rank} | Step {step} | Forward Time: {forward_time:.2f} ms | Backward Time: {backward_time:.2f} ms  | Loss: {loss.item():.4f} | Iteration Time: {iteration_time:.2f} ms | AR Time: {ar_time:.2f} ms | AR Buffer Size: {total_buffer_size / 1024:.2f} KB{reset_color}\n")

            epoch_loss += loss.item()
            nvtx.range_pop()  # Step
            itr += 1
            if itr == 50:
                break

        if rank == 0:
            print(f"[Epoch {epoch+1}] Avg Loss: {epoch_loss / len(train_loader)}")
        nvtx.range_pop()  # Epoch

    cleanup_ddp()

# def run_ddp():
#     world_size = torch.cuda.device_count()
#     mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_ddp(rank, world_size)
