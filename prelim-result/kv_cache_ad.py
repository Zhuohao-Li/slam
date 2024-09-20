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

def visualize_attention_scores(tokenizer, generated_ids, attention_scores):
    # Decode generated tokens
    tokens = [tokenizer.decode(token) for token in generated_ids[0]]
    
    # Prepare attention scores for visualization
    max_len = max(score.size(1) for score in attention_scores)
    padded_scores = []
    for score in attention_scores:
        pad_size = max_len - score.size(1)
        padded_score = torch.nn.functional.pad(score, (0, pad_size))
        padded_scores.append(padded_score)
    
    attention_matrix = torch.stack(padded_scores).squeeze().mean(dim=1).numpy()
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap='YlOrRd')
    plt.title('Attention Score Heatmap')
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def visualize_adjacent_token_attention(tokenizer, generated_ids, attention_scores):
    tokens = [tokenizer.decode(token) for token in generated_ids[0]]
    
    fig, axes = plt.subplots(len(attention_scores) - 1, 1, figsize=(12, 4 * (len(attention_scores) - 1)), squeeze=False)
    fig.suptitle('Adjacent Token Attention Scores', fontsize=16)

    for i in range(len(attention_scores) - 1):
        current_score = attention_scores[i].mean(dim=0).numpy()
        next_score = attention_scores[i+1].mean(dim=0).numpy()
        
        # Ensure both arrays have the same length
        max_len = max(len(current_score), len(next_score))
        current_score = np.pad(current_score, (0, max_len - len(current_score)))
        next_score = np.pad(next_score, (0, max_len - len(next_score)))
        
        ax = axes[i, 0]
        ax.plot(current_score, label=f'Token {i+1}: {tokens[i+1]}')
        ax.plot(next_score, label=f'Token {i+2}: {tokens[i+2]}')
        ax.set_xticks(range(max_len))
        ax.set_xticklabels(tokens[:max_len], rotation=90)
        ax.set_ylabel('Attention Score')
        ax.legend()
        ax.set_title(f'Attention Scores for Tokens {i+1} and {i+2}')

    plt.tight_layout()
    plt.savefig('kv2.png')
    plt.show()

# Load pre-trained model and tokenizer
model = GPT2WithAttentionScores.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text and get attention scores
prompt = "Once upon a time"
generated_ids, attention_scores = generate_with_attention_scores(model, tokenizer, prompt, max_length=20)

# Decode the generated text
generated_text = tokenizer.decode(generated_ids[0])

print("Generated text:", generated_text)
print("Number of attention score tensors:", len(attention_scores))
print("Shape of each attention score tensor:", [score.shape for score in attention_scores])

# Visualize attention scores
visualize_attention_scores(tokenizer, generated_ids, attention_scores)

# Visualize adjacent token attention scores
visualize_adjacent_token_attention(tokenizer, generated_ids, attention_scores)