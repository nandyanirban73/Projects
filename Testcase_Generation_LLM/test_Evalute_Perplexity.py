from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = GPT2TokenizerFast.from_pretrained("./fine_tuned_model")

def calculate_perplexity(sentence):
    encodings = tokenizer(sentence, return_tensors="pt")
    max_length = encodings.input_ids.size(1)
    stride = 512
    nlls = []

    for i in range(0, max_length, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, max_length)
        trg_len = end_loc - i  # Length of target sequence
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

# Test sentence
sentence = "The capital of India is New Delhi."
print(f"Perplexity: {calculate_perplexity(sentence)}")
