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

#print(df_csv.head())
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

#print(df_csv.head())




# Read a .py file
py_file = "C:/Users/nandy/OneDrive/Documents/Visual_Studio_Workspace/Python_Workspace/Testcase_Generation_LLM/IoT_App/IoT_without_infi_loop.py"

with open(py_file, "r") as file:
    code_content = file.read()

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
        "code": code_content,  # Assuming the same code matches all requirements
        "test_cases": related_test_cases_data
    }
    
    dataset.append(entry)

df_dataset = pd.DataFrame(dataset)
df_dataset.to_csv("test_dataset_output.csv", index = False)
print("dataset prepared!")