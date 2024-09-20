import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class GPT2WithAttentionScores(GPT2LMHeadModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        output = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_attentions=True,
            **kwargs
        )
        return output

def generate_with_attention_scores(model, tokenizer, prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    past_key_values = None
    generated = input_ids
    attention_scores = []

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, past_key_values=past_key_values)
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        generated = torch.cat([generated, next_token], dim=1)
        input_ids = next_token
        past_key_values = outputs.past_key_values

        # Get attention scores for the newly generated token
        attention_scores.append(outputs.attentions[-1][:, :, -1, :].squeeze())

    return generated, attention_scores

def visualize_top_40_attention_scores(tokenizer, generated_ids, attention_scores):
    tokens = [tokenizer.decode(token) for token in generated_ids[0]]
    
    # Get the last two generated tokens' attention scores
    last_two_scores = attention_scores[-2:]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Top 40 Tokens with Highest Attention Scores for Last Two Generated Tokens', fontsize=16)
    
    for i, scores in enumerate(last_two_scores):
        # Average attention scores across heads
        avg_scores = scores.mean(dim=0).numpy()
        
        # Get the top 40 indices
        top_40_indices = np.argsort(avg_scores)[-40:][::-1]
        
        # Get the corresponding tokens and scores
        top_40_tokens = [tokens[idx] for idx in top_40_indices]
        top_40_scores = avg_scores[top_40_indices]
        
        # Plot
        ax = axes[i]
        bars = ax.barh(range(len(top_40_tokens)), top_40_scores)
        ax.set_yticks(range(len(top_40_tokens)))
        ax.set_yticklabels(top_40_tokens)
        ax.set_title(f'Token {len(tokens)-1+i}: "{tokens[-2+i]}"')
        ax.set_xlabel('Average Attention Score')
        ax.invert_yaxis()  # To have the highest score at the top
        
        # Highlight tokens from KV cache
        for j, (token, bar) in enumerate(zip(top_40_tokens, bars)):
            if top_40_indices[j] < len(avg_scores) - 1:  # If the token was in KV cache
                bar.set_color('red')
            ax.get_yticklabels()[j].set_color('black')  # Set all labels to black
        
        ax.set_xlim(0, max(top_40_scores) * 1.1)  # Set x-axis limit with some padding
    
    plt.tight_layout()
    plt.savefig('kv6.png')
    plt.show()

def visualize_top_40_attention_scores_indices(tokenizer, generated_ids, attention_scores):
    tokens = [tokenizer.decode(token) for token in generated_ids[0]]
    
    # Get the last two generated tokens' attention scores
    last_two_scores = attention_scores[-2:]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Top 40 Tokens with Highest Attention Scores for Last Two Generated Tokens', fontsize=16)
    
    # Get the average attention scores and sort indices for both last tokens
    avg_scores_1 = last_two_scores[0].mean(dim=0).numpy()
    avg_scores_2 = last_two_scores[1].mean(dim=0).numpy()
    
    top_40_indices_1 = np.argsort(avg_scores_1)[-40:][::-1]
    top_40_indices_2 = np.argsort(avg_scores_2)[-40:][::-1]
    
    common_indices = set(top_40_indices_1).intersection(set(top_40_indices_2))
    
    for i, (scores, top_40_indices) in enumerate(zip(last_two_scores, [top_40_indices_1, top_40_indices_2])):
        ax = axes[i]
        avg_scores = scores.mean(dim=0).numpy()
        top_40_scores = avg_scores[top_40_indices]
        
        bars = ax.barh(range(len(top_40_indices)), top_40_scores)
        ax.set_yticks(range(len(top_40_indices)))
        ax.set_yticklabels(top_40_indices)
        ax.set_title(f'Token {len(tokens)-2+i}: Index {generated_ids[0][-2+i].item()}')
        ax.set_xlabel('Average Attention Score')
        ax.invert_yaxis()

        # Highlight common indices in both sets
        for j, (index, bar) in enumerate(zip(top_40_indices, bars)):
            if index in common_indices:
                bar.set_color('red')
                ax.get_yticklabels()[j].set_color('red')  # Highlight common indices in red

        ax.set_xlim(0, max(top_40_scores) * 1.1)
    
    plt.tight_layout()
    plt.savefig('kv6_indices.png')
    plt.show()

def visualize_top_10_to_50_attention_scores_indices(tokenizer, generated_ids, attention_scores):
    tokens = [tokenizer.decode(token) for token in generated_ids[0]]
    
    # Get the last two generated tokens' attention scores
    last_two_scores = attention_scores[-2:]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 20))
    fig.suptitle('Tokens Ranked 10th to 50th with Highest Attention Scores for Last Two Generated Tokens\n', fontsize=16)
    
    # Get the average attention scores and sort indices for both last tokens
    avg_scores_1 = last_two_scores[0].mean(dim=0).numpy()
    avg_scores_2 = last_two_scores[1].mean(dim=0).numpy()
    
    # Get the top 50 indices, then filter out the top 10 to get tokens ranked from 10 to 50
    top_50_indices_1 = np.argsort(avg_scores_1)[-50:][::-1][10:]
    top_50_indices_2 = np.argsort(avg_scores_2)[-50:][::-1][10:]
    
    # Find common indices between both sets
    common_indices = set(top_50_indices_1).intersection(set(top_50_indices_2))
    
    # Count the number of common tokens
    num_common_tokens = len(common_indices)
    print(f"Number of common tokens between the last two generated tokens: {num_common_tokens}")
    
    for i, (scores, top_50_indices) in enumerate(zip(last_two_scores, [top_50_indices_1, top_50_indices_2])):
        ax = axes[i]
        avg_scores = scores.mean(dim=0).numpy()
        top_50_scores = avg_scores[top_50_indices]
        
        bars = ax.barh(range(len(top_50_indices)), top_50_scores)
        ax.set_yticks(range(len(top_50_indices)))
        ax.set_yticklabels(top_50_indices)
        ax.set_title(f'Token {len(tokens)-2+i}')
        ax.set_xlabel('Average Attention Score')
        ax.invert_yaxis()

        # Highlight common indices in both sets
        for j, (index, bar) in enumerate(zip(top_50_indices, bars)):
            if index in common_indices:
                bar.set_color('#FF6B6B')
                ax.get_yticklabels()[j].set_color('#FF6B6B')  # Highlight common indices in #FF6B6B

        ax.set_xlim(0, max(top_50_scores) * 1.1)
    
    plt.tight_layout()
    plt.savefig('kv6_indices_top10_to_50.png')
    plt.show()




# Load pre-trained model and tokenizer
model = GPT2WithAttentionScores.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text and get attention scores
prompt = "How is the ground truth for fake news established? Characterizing Political Fake News in Twitter by its Meta-DataJulio Amador Díaz LópezAxel Oehmichen Miguel Molina-Solana( j.amador, axelfrancois.oehmichen11, mmolinas@imperial.ac.uk ) Imperial College London This article presents a preliminary approach towards characterizing political fake news on Twitter through the analysis of their meta-data. In particular, we focus on more than 1.5M tweets collected on the day of the election of Donald Trump as 45th president of the United States of America. We use the meta-data embedded within those tweets in order to look for differences between tweets containing fake news and tweets not containing them. Specifically, we perform our analysis only on tweets that went viral, by studying proxies for users' exposure to the tweets, by characterizing accounts spreading fake news, and by looking at their polarization. We found significant differences on the distribution of followers, the number of URLs on tweets, and the verification of the users."

generated_ids, attention_scores = generate_with_attention_scores(model, tokenizer, prompt, max_length=50)

# Decode the generated text
generated_text = tokenizer.decode(generated_ids[0])

print("Generated text:", generated_text)
print("Number of attention score tensors:", len(attention_scores))
print("Shape of each attention score tensor:", [score.shape for score in attention_scores])

# Visualize top 50 attention scores for the last two generated tokens
# Use the updated function
visualize_top_10_to_50_attention_scores_indices(tokenizer, generated_ids, attention_scores)