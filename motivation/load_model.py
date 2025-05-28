from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model name
model_name = "meta-llama/Llama-3.2-3B"  # Replace with the LLaMA 2 model name you want
model_short_name = model_name.split("/")[-1]

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the tokenizer and model locally
tokenizer.save_pretrained(f"./{model_short_name}")
model.save_pretrained(f"./{model_short_name}")

print("Model and tokenizer saved!")
