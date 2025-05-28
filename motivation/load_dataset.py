from datasets import load_dataset
from transformers import AutoTokenizer

# Load Alpaca dataset (it's a DatasetDict with only 'train' split)
dataset_dict = load_dataset("tatsu-lab/alpaca")
dataset = dataset_dict["train"]  # <-- Access the actual dataset

dataset.save_to_disk("alpaca")

