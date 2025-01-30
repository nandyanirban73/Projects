from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
import faiss
import numpy as np
import pandas as pd
import json
from datasets import load_dataset
from datasets import Dataset

#Importing existing data in vector database
# Load .csv file
Req_xlsx_file = "C:/Users/nandy/OneDrive/Documents/Visual_Studio_Workspace/Python_Workspace/Testcase_Generation_LLM/IoT_App/Requirements.xlsx"
tc_csv_file = "C:/Users/nandy/OneDrive/Documents/Visual_Studio_Workspace/Python_Workspace/Testcase_Generation_LLM/IoT_App/testcases.csv"
trace_xlsx_file = "C:/Users/nandy/OneDrive/Documents/Visual_Studio_Workspace/Python_Workspace/Testcase_Generation_LLM/IoT_App/traceability.xlsx"
df_csv = pd.read_csv(tc_csv_file,encoding="ISO-8859-1")
df_xlsx =pd.read_excel(Req_xlsx_file)
df_trace = pd.read_excel(trace_xlsx_file)# Load the traceability information

#print(df_xlsx.columns.to_list())
#print(df_xlsx.head())

#Clean and Normalize Data
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = " ".join(text.split())
    return text

# Normalize data
# Apply normalize_text to all columns with data
for column in df_csv.columns:
    # Check if the column contains string data
    if df_csv[column].dtype == "object":  # 'object' usually means text data
        df_csv[column] = df_csv[column].apply(normalize_text)


# Read a .py file
#py_file = "C:/Users/nandy/OneDrive/Documents/Visual_Studio_Workspace/Python_Workspace/Testcase_Generation_LLM/IoT_App/IoT_without_infi_loop.py"

#with open(py_file, "r") as file:
#    code_content = file.read()

#print(code_content)

# Create a unified dataset
dataset = []

for _, row in df_xlsx.iterrows():

    # Extract the requirement
    requirement = row["Requirement"]
    
    # Find associated test cases using traceability mapping

    # Extract and split the TestCaseID column into individual IDs
    related_test_cases_ids = (
    df_trace[df_trace["Requirement"] == requirement]["TestCaseID"]
    .str.split(",")  # Split comma-separated values into lists
    .explode()       # Flatten the resulting lists into a single series
    .str.strip()     # Remove any leading/trailing spaces
    .tolist()        # Convert back to a list
    )

    # Extract the actual test case rows based on IDs. Converting it to lower case to avoid mismatch
    related_test_cases = df_csv[df_csv["TestCaseID"].str.lower().isin([id.lower() for id in related_test_cases_ids])]

    # Convert the related test case rows to a list of dictionaries (each row as a dictionary)
    related_test_cases_data = related_test_cases.to_dict(orient="records")

    # Create an entry for the dataset
    entry = {
        "requirement": requirement,
        #"code": code_content,  # Assuming the same code matches all requirements
        "test_cases": related_test_cases_data
    }
    
    dataset.append(entry)

# Save the dataset to a JSON file
with open("prepared_dataset.json", "w") as file:
    json.dump(dataset, file, indent=4)

print("Dataset prepared and saved to prepared_dataset.json")

# Load tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load the prepared dataset
with open("prepared_dataset.json", "r") as file:
    dataset = json.load(file)

# Initialize a sentence transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and efficient

# Prepare the documents and their embeddings
documents = []
embeddings = []

for entry in dataset:
    # Combine requirement, code, and test cases into a single string
    combined_text = entry["requirement"]
    for test_case in entry["test_cases"]:
        combined_text += " " + " ".join(f"{key}: {value}" for key, value in test_case.items())
    
    documents.append(combined_text)
    embeddings.append(embedding_model.encode(combined_text))

# Convert embeddings to NumPy array
embeddings = np.array(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
index.add(embeddings)  # Add embeddings to the index

# Save the FAISS index
faiss.write_index(index, "vector_database.faiss")

# Save the documents for retrieval reference
with open("documents.json", "w") as file:
    json.dump(documents, file, indent=4)

print("Vector database created and saved.")



ENABLE_MODEL_TRAINING = False
if ENABLE_MODEL_TRAINING :

    # Load the tokenizer and model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure that the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load and preprocess the dataset
    def preprocess_data(json_path):
        """Load and preprocess the dataset from a JSON file."""
        with open(json_path, "r") as file:
            data = json.load(file)
        
        examples = []
        for entry in data:
            prompt = (
                f"Requirement: {entry['requirement']}\n"
                #f"Code: {entry['code']}\n"
                #f"Test Cases: {', '.join(entry['test_cases'])}\n"
                f"Test Cases: {entry['test_cases']}"
                f"Output:"
            )
            examples.append({"input": prompt, "output": entry["requirement"]})
        
        return Dataset.from_list(examples)

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize the input-output pairs."""
        inputs = tokenizer(
            examples["input"], 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )
        outputs = tokenizer(
            examples["output"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        inputs["labels"] = outputs["input_ids"]

        # Ensure the labels have the same padding as input_ids
        labels = inputs["labels"]
        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels
        ]
        inputs["labels"] = labels
        return inputs

    # Load the dataset
    dataset = preprocess_data("prepared_dataset.json")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input", "output"])

    # Use a data collator for dynamic padding
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False because we're working with causal language modeling
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model_2",  # Output directory for saving the model
        overwrite_output_dir=True,
        num_train_epochs=3,              # Number of training epochs
        per_device_train_batch_size=4,   # Batch size per GPU/CPU
        save_steps=500,                  # Save checkpoint every 500 steps
        save_total_limit=2,              # Limit on checkpoints
        logging_dir="./logs",            # Directory for logs
        logging_steps=100,
        evaluation_strategy="no",        # Set to "steps" if validation is used
        learning_rate=5e-5,              # Learning rate
        weight_decay=0.01,               # Weight decay
        warmup_steps=100,                # Warmup steps
        fp16=True,                       # Mixed precision for faster training on GPUs
    )

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Dynamic padding during training
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model_2")
    tokenizer.save_pretrained("./fine_tuned_model_2")
    print("Fine-tuning complete. Model saved at './fine_tuned_model_2'.")








# Load the fine-tuned model
model_name = "./fine_tuned_model_2"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model Loaded")

# Load the FAISS index and documents
index = faiss.read_index("vector_database.faiss")
with open("documents.json", "r") as file:
    documents = json.load(file)

print("FAISS index and documents Loaded")


# Retrieve Context
def retrieve_context(query, top_k=3):
    """Retrieve relevant documents for the query."""
    query_embedding = embedding_model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    # Combine and truncate the context
    combined_context = " ".join(retrieved_docs)
    #if len(combined_context) > max_length:
    #    combined_context = combined_context[:max_length]  # Truncate context
    return combined_context


# Define RAG pipeline. Generate Response by sending the query and related document to LLM
def generate_output(query):
    """Generate updated requirements, code, and test cases."""
    context = retrieve_context(query)

    # Truncate input if necessary
    max_input_length = 1024  # Adjust based on model's capability
    if len(context) > max_input_length:
        context = context[:max_input_length]
    input_prompt = f"Context: {context}\nQuery: {query}\nOutput:"

    # Encode input with attention mask
    inputs = tokenizer(
        input_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_input_length
    )

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Ensure attention mask is used
        max_new_tokens=200,  # Adjust based on your requirements
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad_token_id
        num_beams=5,
        #early_stopping=True
        temperature=0.8,  # Adjust temperature for creativity
        top_k=50,         # Use top-k sampling
        top_p=0.9         # Use nucleus sampling
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the RAG pipeline
query = "Write down the test cases when the temperature is 8°C. in night 11 pm and the month is December"
result = generate_output(query)
print("Generated Output:")
print(result)