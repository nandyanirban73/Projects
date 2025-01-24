from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline

# Load tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset and add labels
def tokenize_function(examples):
    # Tokenize with padding and truncation
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # Clone input_ids for labels
    inputs["labels"] = inputs["input_ids"].copy()
    # Set padding tokens in labels to -100 to ignore during loss computation
    inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label] 
        for label in inputs["labels"]
    ]
    return inputs


# Load your dataset
dataset = load_dataset("text", data_files="dataset.txt")


# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Load GPT-2 model
model = AutoModelForCausalLM.from_pretrained(model_name)


# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Where to save the model
    overwrite_output_dir=True,
    num_train_epochs=3,              # Number of epochs
    per_device_train_batch_size=4,   # Batch size
    save_steps=500,                  # Save checkpoint every 500 steps
    save_total_limit=2,              # Keep only the last 2 checkpoints
    logging_dir="./logs",            # Log directory
    logging_steps=100,
    evaluation_strategy="no",        # Set to "steps" if you have validation data
    learning_rate=5e-5,              # Learning rate
    weight_decay=0.01,               # Weight decay for optimizer
    warmup_steps=100,                # Warmup steps
    fp16=True,                       # Mixed precision for faster training on GPUs
)


# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)


#Start Training
trainer.train()

# Save the Fine tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")



 