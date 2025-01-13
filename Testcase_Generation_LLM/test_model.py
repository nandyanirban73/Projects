from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a pre-trained model
model_name = "gpt2"  # A lightweight model for testing
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test generation
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)

# Decode and print output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
