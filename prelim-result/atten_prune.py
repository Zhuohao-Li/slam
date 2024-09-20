import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# Initialize the model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to run inference
def run_inference(input_text, prune_cache=False):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        # Forward pass to get logits
        outputs = model(input_ids, use_cache=True, return_dict=True)
        logits = outputs.logits
        if prune_cache:
            # Example pruning: Zero out the first half of the model's parameters
            with torch.no_grad():
                for param in model.parameters():
                    param.data[param.data.shape[0]//2:] = 0
        return logits

# Example text
# input_text = "A is B, B is C, C is A, A is B, B is C, C is A, A is "
input_text = "How is the ground truth for fake news established? Characterizing Political Fake News in Twitter by its Meta-DataJulio Amador Díaz LópezAxel Oehmichen Miguel Molina-Solana( j.amador, axelfrancois.oehmichen11, mmolinas@imperial.ac.uk ) Imperial College London This article presents a preliminary approach towards characterizing political fake news on Twitter through the analysis of their meta-data. In particular, we focus on more than 1.5M tweets collected on the day of the election of Donald Trump as 45th president of the United States of America. We use the meta-data embedded within those tweets in order to look for differences between tweets containing fake news and tweets not containing them. Specifically, we perform our analysis only on tweets that went viral, by studying proxies for users' exposure to the tweets, by characterizing accounts spreading fake news, and by looking at their polarization. We found significant differences on the distribution of followers, the number of URLs on tweets, and the verification of the users.\n]\nIntroduction\nWhile fake news, understood as deliberately misleading pieces of information, have existed since long ago (e.g. it is not unusual to receive news falsely claiming the death of a celebrity), the term reached the mainstream, particularly so in politics, during the 2016 presidential election in the United States BIBREF0 . Since then, governments and corporations alike (e.g. Google BIBREF1 and Facebook BIBREF2 ) have begun efforts to tackle fake news as they can affect political decisions BIBREF3 . Yet, the ability to define, identify and stop fake news from spreading is limited.\nSince the Obama campaign in 2008, social media has been pervasive in the political arena in the United States. Studies report that up to 62% of American adults receive their news from social media BIBREF4 . The wide use of platforms such as Twitter and Facebook has facilitated the diffusion of fake news by simplifying the process of receiving content with no significant third party filtering, fact-checking or editorial judgement. Such characteristics make these platforms suitable means for sharing news that, disguised as legit ones, try to confuse readers.\nSuch use and their prominent rise has been confirmed by Craig Silverman, a Canadian journalist who is a prominent figure on fake news BIBREF5 : “In the final three months of the US presidential campaign, the top-performing fake election news stories on Facebook generated more engagement than the top stories from major news outlet”.\nOur current research hence departs from the assumption that social media is a conduit for fake news and asks the question of whether fake news (as spam was some years ago) can be identified, modelled and eventually blocked. In order to do so, we use a sample of more that 1.5M tweets collected on November 8th 2016 —election day in the United States— with the goal of identifying features that tweets containing fake news are likely to have. As such, our paper aims to provide a preliminary characterization of fake news in Twitter by looking into meta-data embedded in tweets. Considering meta-data as a relevant factor of analysis is in line with findings reported by Morris et al. BIBREF6 . We argue that understanding differences between tweets containing fake news and regular tweets will allow researchers to design mechanisms to block fake news in Twitter.\nSpecifically, our goals are: 1) compare the characteristics of tweets labelled as containing fake news to tweets labelled as not containing them, 2) characterize, through their meta-data, viral tweets containing fake news and the accounts from which they originated, and 3) determine the extent to which tweets containing fake news expressed polarized political views.\nFor our study, we used the number of retweets to single-out those that went viral within our sample. Tweets within that subset (viral tweets hereafter) are varied and relate to different topics. We consider that a tweet contains fake news if its text falls within any of the following categories described by Rubin et al. BIBREF7 (see next section for the details of such categories): serious fabrication, large-scale hoaxes, jokes taken at face value, slanted reporting of real facts and stories where the truth is contentious. The dataset BIBREF8 , manually labelled by an expert, has been publicly released and is available to researchers and interested parties.\nFrom our results, the following main observations can be made:\nOur findings resonate with similar work done on fake news such as the one from Allcot and Gentzkow BIBREF9 . Therefore, even if our study is a preliminary attempt at characterizing fake news on Twitter using only their meta-data, our results provide external validity to previous research. Moreover, our work not only stresses the importance of using meta-data, but also underscores which parameters may be useful to identify fake news on Twitter.\nThe rest of the paper is organized as follows. "

# Run inference without pruning
logits_no_prune = run_inference(input_text, prune_cache=False)

# Run inference with pruning
logits_prune = run_inference(input_text, prune_cache=True)

# Convert logits to probabilities
prob_no_prune = torch.nn.functional.softmax(logits_no_prune, dim=-1).squeeze(0).detach().numpy()
prob_prune = torch.nn.functional.softmax(logits_prune, dim=-1).squeeze(0).detach().numpy()

# Get token IDs and convert to tokens
input_ids = tokenizer.encode(input_text, return_tensors='pt')[0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Check if probabilities match the tokens length
print(f'Tokens: {tokens}')
print(f'Probabilities shape (without pruning): {prob_no_prune.shape}')
print(f'Probabilities shape (with pruning): {prob_prune.shape}')

# Ensure that the length of tokens matches the dimensions of the probabilities
if prob_no_prune.shape[0] != len(tokens) or prob_prune.shape[0] != len(tokens):
    raise ValueError("Shape mismatch between tokens and probabilities")

# Create a combined bar chart to compare logits distributions
def plot_comparison(prob_no_prune, prob_prune, tokens):
    plt.figure(figsize=(14, 8))

    x = np.arange(len(tokens))
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width/2, prob_no_prune[:, 0], width, label='Without Pruning')
    bars2 = ax.bar(x + width/2, prob_prune[:, 0], width, label='With Pruning')

    ax.set_xlabel('Tokens')
    ax.set_ylabel('Probabilities')
    ax.set_title('Token Probabilities Comparison Before and After Pruning')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('comparison_probabilities.png')
    plt.show()

# Plot the comparison
plot_comparison(prob_no_prune, prob_prune, tokens)
