import matplotlib.pyplot as plt
import torch
from transformers import GPT2Model, GPT2Tokenizer
from collections import defaultdict
import numpy as np

def analyze_attention(input_text, model_name='gpt2', top_n=10, output_file='attention_analysis.txt'):
    # Initialize model and tokenizer
    model = GPT2Model.from_pretrained(model_name, output_attentions=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate outputs with attention scores
    outputs = model(input_ids)

    # Extract attention scores from the last two layers
    last_layer_attentions = outputs.attentions[-1]
    second_last_layer_attentions = outputs.attentions[-2]

    # Take the average attention scores across all heads for both layers
    avg_attention_last = last_layer_attentions.mean(dim=1).squeeze().detach().numpy()
    avg_attention_second_last = second_last_layer_attentions.mean(dim=1).squeeze().detach().numpy()

    # Convert token ids back to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Process attention scores for both layers
    layers = [('last', avg_attention_last), ('second_last', avg_attention_second_last)]
    results = {}

    for layer_name, avg_attention in layers:
        top_token_frequency = defaultdict(int)
        top_token_scores = defaultdict(list)

        for i, token in enumerate(tokens):
            attention_scores = avg_attention[i, :]
            top_indices = np.argsort(attention_scores)[-top_n:][::-1]
            top_values = attention_scores[top_indices]

            for idx, value in zip(top_indices, top_values):
                top_token_frequency[tokens[idx]] += 1
                top_token_scores[tokens[idx]].append(value)

        avg_scores = {token: np.mean(scores) for token, scores in top_token_scores.items()}
        results[layer_name] = (avg_scores, top_token_frequency)

    return tokens, results

def visualize_top_10_to_40(tokens, results, output_file='compare.png'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Get the top 10 to 40 tokens for both layers
    top_10_to_40_tokens = {}
    for layer_name, (avg_scores, top_token_frequency) in results.items():
        sorted_tokens = sorted(avg_scores.keys(), key=lambda x: (-top_token_frequency[x], -avg_scores[x]))
        top_10_to_40_tokens[layer_name] = set(sorted_tokens[9:40])

    # Find common tokens
    common_tokens = top_10_to_40_tokens['last'].intersection(top_10_to_40_tokens['second_last'])
    print(len(common_tokens))

    for idx, (layer_name, (avg_scores, top_token_frequency)) in enumerate(results.items()):
        # Sort tokens by frequency and then by average score
        sorted_tokens = sorted(avg_scores.keys(), key=lambda x: (-top_token_frequency[x], -avg_scores[x]))
        top_10_to_40_tokens_list = sorted_tokens[9:40]  # Get tokens ranked 10th to 40th

        # Find indices for these tokens
        token_to_index = {token: i for i, token in enumerate(tokens)}
        top_10_to_40_indices = [token_to_index[token] for token in top_10_to_40_tokens_list]

        # Visualization for tokens ranked 10th to 40th
        ax = ax1 if idx == 0 else ax2
        bars = ax.bar(range(len(top_10_to_40_indices)), 
                      [avg_scores[token] for token in top_10_to_40_tokens_list],
                      color=['#FF6B6B' if token in common_tokens else '#1FD665' for token in top_10_to_40_tokens_list])

        ax.set_xlabel('Token Indices (Ranked 10th to 40th by Frequency)')
        ax.set_ylabel('Average Attention Scores')
        ax.set_title(f'{layer_name.capitalize()} Layer: Tokens Ranked 10th to 40th by Frequency in Top Attention Scores')
        ax.set_xticks(range(len(top_10_to_40_indices)))
        ax.set_xticklabels(top_10_to_40_indices, rotation=45, ha='right')

        # Add token and frequency labels on top of each bar
        for i, (bar, (index, token)) in enumerate(zip(bars, zip(top_10_to_40_indices, top_10_to_40_tokens_list))):
            height = bar.get_height()
            # ax.text(bar.get_x() + bar.get_width()/2., height,
            #         f'{token}\n(freq: {top_token_frequency[token]})',
            #         ha='center', va='bottom', rotation=40, fontsize=8)

    # Add a legend
    fig.legend(['Common to both layers', 'Unique to layer'], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved to {output_file}")

def main():
    input_text = "How is the ground truth for fake news established? Characterizing Political Fake News in Twitter by its Meta-DataJulio Amador Díaz LópezAxel Oehmichen Miguel Molina-Solana( j.amador, axelfrancois.oehmichen11, mmolinas@imperial.ac.uk ) Imperial College London This article presents a preliminary approach towards characterizing political fake news on Twitter through the analysis of their meta-data. In particular, we focus on more than 1.5M tweets collected on the day of the election of Donald Trump as 45th president of the United States of America. We use the meta-data embedded within those tweets in order to look for differences between tweets containing fake news and tweets not containing them. Specifically, we perform our analysis only on tweets that went viral, by studying proxies for users' exposure to the tweets, by characterizing accounts spreading fake news, and by looking at their polarization. We found significant differences on the distribution of followers, the number of URLs on tweets, and the verification of the users."

    tokens, results = analyze_attention(input_text)
    visualize_top_10_to_40(tokens, results)

if __name__ == "__main__":
    main()