# sample RAG workflow
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np

# Sample documents
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "India's Capital is New Delhi"
]

# Generate embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = embedder.encode(documents)

# Index embeddings using FAISS
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(np.array(document_embeddings))

# Save the index for later use
faiss.write_index(index, "document_index")

# Load FAISS index and documents
index = faiss.read_index("document_index")
query = "What is the capital of India?"
query_embedding = embedder.encode([query])

# Retrieve the closest document
D, I = index.search(np.array(query_embedding), k=1)  # k is the number of results
retrieved_doc = documents[I[0][0]]
print(f"Retrieved Document: {retrieved_doc}")

# Load a pre-trained model
model_name = "gpt2"  # A lightweight model for testing
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use the retrieved document as context
context = f"Context: {retrieved_doc}\nQuestion: {query}\nAnswer:"
inputs = tokenizer(context, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
