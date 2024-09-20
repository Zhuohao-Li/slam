from transformers import OPTForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pardata


# Load the pre-trained OPT model
model_name = "facebook/opt-125m"
model = OPTForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# wiki-text-103
data = pardata.load_dataset('wikitext103')
# print(data[0])

""" longbench dataset
# load the dataset from HF repo
datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
    print(data)

# validaiton_texts
"""
# conduct zero-shot inference
def generate_attention_scores(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    return outputs.attentions
    
# Generate attention scores for each sentence in the validation set
all_attention_scores = [generate_attention_scores(text, model, tokenizer) for text in data]

print(all_attention_scores)

def calculate_sparsity(attention_scores):
    sparsity = []
    for layer_attention in attention_scores:
        # Assume attention_scores is a list of attention tensors for each layer
        # For simplicity, use the average sparsity across all attention heads in a layer
        layer_sparsity = []
        for attention_matrix in layer_attention:
            non_zero_elements = (attention_matrix > 1e-5).sum().item()
            total_elements = attention_matrix.numel()
            sparsity_value = 1 - (non_zero_elements / total_elements)
            layer_sparsity.append(sparsity_value)
        sparsity.append(np.mean(layer_sparsity))
    return sparsity

# Calculate sparsity for each layer
layer_sparsities = [calculate_sparsity(attentions) for attentions in all_attention_scores]

print(layer_sparsities)

# Plot layer-wise sparsity
plt.figure(figsize=(10, 6))
for i, sparsity in enumerate(layer_sparsities):
    plt.plot(sparsity, label=f'Sample {i+1}')
plt.xlabel("Layer")
plt.ylabel("Sparsity")
plt.title("Layer-wise Sparsity in Attention Blocks")
plt.legend()
plt.show()

# Visualize Normalized Attention Scores
def plot_attention_scores(attention_scores, token_list, layer_idx=0):
    attention_matrix = attention_scores[layer_idx][0].mean(dim=0).detach().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap='viridis')
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(token_list)), labels=token_list, rotation=90)
    plt.yticks(ticks=np.arange(len(token_list)), labels=token_list)
    plt.title(f'Layer {layer_idx+1} Attention Scores')
    plt.show()

# Visualize the attention scores for the first sample in the validation set
# sample_attention_scores = all_attention_scores[0]
# tokens = tokenizer.tokenize(validation_texts[0])
# plot_attention_scores(sample_attention_scores, tokens)


