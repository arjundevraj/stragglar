import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
import torch.cuda.nvtx as nvtx
from datasets import load_from_disk

rank_colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
reset_color = "\033[0m"

def setup_ddp(rank, world_size):
    dev_id = torch.device('cuda', int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=dev_id)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    color = rank_colors[rank % len(rank_colors)]
    print(f"{color}Rank {rank} | device_id: {dev_id}")

def cleanup_ddp():
    dist.destroy_process_group()

def format_example(example, tokenizer):
    prompt = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input"):
        prompt += f"### Input:\n{example['input']}\n\n"
    prompt += f"### Response:\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

def train_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model_path = "./Llama-3.2-3B"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token

    # Load from local alpaca dataset
    dataset_path = "./alpaca"  # Path to local dataset directory
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path).select(range(8192))
    else:
        raise FileNotFoundError(f"Local Alpaca dataset not found at {dataset_path}")
    train_dataset = dataset

    train_dataset = train_dataset.map(lambda ex: format_example(ex, tokenizer), remove_columns=train_dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    EPOCH = 1
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_loader) * EPOCH
    )

    itr = 0
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
            loss = outputs.loss


            torch.cuda.synchronize()
            nvtx.range_pop()
            forward_end_time.record()

            backward_start_time.record()
            nvtx.range_push("Backward")
            loss.backward()
            torch.cuda.synchronize()
            nvtx.range_pop()
            backward_end_time.record()

            nvtx.range_push("AllReduce_Gradients")
            total_buffer_size = 0
            ar_start_time.record()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    nvtx.range_push(f"allreduce:{name}")
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
                    nvtx.range_pop()
                    # Record buffer size for communication
                    buffer_size = param.grad.data.numel() * param.grad.data.element_size()
                    total_buffer_size += buffer_size
                    
            torch.cuda.synchronize()
            ar_end_time.record()
            nvtx.range_pop()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            iteration_end_time.record()
            torch.cuda.synchronize()

            forward_time = forward_start_time.elapsed_time(forward_end_time)
            backward_time = backward_start_time.elapsed_time(backward_end_time)
            iteration_time = iteration_start_time.elapsed_time(iteration_end_time)
            ar_time = ar_start_time.elapsed_time(ar_end_time)

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

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_ddp(rank, world_size)
