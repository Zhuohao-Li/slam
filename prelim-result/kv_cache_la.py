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

def visualize_kv_cache_overlap(tokenizer, generated_ids, attention_scores):
    tokens = [tokenizer.decode(token) for token in generated_ids[0]]
    
    # Get the last two generated tokens' attention scores
    last_two_scores = attention_scores[-2:]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, scores in enumerate(last_two_scores):
        # Average attention scores across heads
        avg_scores = scores.mean(dim=0).numpy()
        
        # Pad or truncate to match the number of tokens
        if len(avg_scores) < len(tokens):
            avg_scores = np.pad(avg_scores, (0, len(tokens) - len(avg_scores)))
        else:
            avg_scores = avg_scores[:len(tokens)]
        
        ax.plot(avg_scores, label=f'Token {len(tokens)-1+i}: "{tokens[-2+i]}"', 
                marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Highlight the overlap region
    overlap_start = len(tokens) - len(last_two_scores[0])
    ax.axvspan(overlap_start, len(tokens)-1, alpha=0.2, color='yellow', label='KV Cache Overlap')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel('Average Attention Score')
    ax.set_title('Attention Scores for Last Two Generated Tokens with KV Cache Overlap')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('kv4.png')
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

# Visualize KV cache overlap for the last two generated tokens
visualize_kv_cache_overlap(tokenizer, generated_ids, attention_scores)