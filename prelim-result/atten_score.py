import matplotlib.pyplot as plt
import torch
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np

# Initialize model and tokenizer
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode the input text
input_text = "How is the ground truth for fake news established? Characterizing Political Fake News in Twitter by its Meta-DataJulio Amador Díaz LópezAxel Oehmichen Miguel Molina-Solana( j.amador, axelfrancois.oehmichen11, mmolinas@imperial.ac.uk ) Imperial College London This article presents a preliminary approach towards characterizing political fake news on Twitter through the analysis of their meta-data. In particular, we focus on more than 1.5M tweets collected on the day of the election of Donald Trump as 45th president of the United States of America. We use the meta-data embedded within those tweets in order to look for differences between tweets containing fake news and tweets not containing them. Specifically, we perform our analysis only on tweets that went viral, by studying proxies for users' exposure to the tweets, by characterizing accounts spreading fake news, and by looking at their polarization. We found significant differences on the distribution of followers, the number of URLs on tweets, and the verification of the users.\n]\nIntroduction\nWhile fake news, understood as deliberately misleading pieces of information, have existed since long ago (e.g. it is not unusual to receive news falsely claiming the death of a celebrity), the term reached the mainstream, particularly so in politics, during the 2016 presidential election in the United States BIBREF0 . Since then, governments and corporations alike (e.g. Google BIBREF1 and Facebook BIBREF2 ) have begun efforts to tackle fake news as they can affect political decisions BIBREF3 . Yet, the ability to define, identify and stop fake news from spreading is limited.\nSince the Obama campaign in 2008, social media has been pervasive in the political arena in the United States. Studies report that up to 62% of American adults receive their news from social media BIBREF4 . The wide use of platforms such as Twitter and Facebook has facilitated the diffusion of fake news by simplifying the process of receiving content with no significant third party filtering, fact-checking or editorial judgement. Such characteristics make these platforms suitable means for sharing news that, disguised as legit ones, try to confuse readers.\nSuch use and their prominent rise has been confirmed by Craig Silverman, a Canadian journalist who is a prominent figure on fake news BIBREF5 : “In the final three months of the US presidential campaign, the top-performing fake election news stories on Facebook generated more engagement than the top stories from major news outlet”.\nOur current research hence departs from the assumption that social media is a conduit for fake news and asks the question of whether fake news (as spam was some years ago) can be identified, modelled and eventually blocked. In order to do so, we use a sample of more that 1.5M tweets collected on November 8th 2016 —election day in the United States— with the goal of identifying features that tweets containing fake news are likely to have. As such, our paper aims to provide a preliminary characterization of fake news in Twitter by looking into meta-data embedded in tweets. Considering meta-data as a relevant factor of analysis is in line with findings reported by Morris et al. BIBREF6 . We argue that understanding differences between tweets containing fake news and regular tweets will allow researchers to design mechanisms to block fake news in Twitter.\nSpecifically, our goals are: 1) compare the characteristics of tweets labelled as containing fake news to tweets labelled as not containing them, 2) characterize, through their meta-data, viral tweets containing fake news and the accounts from which they originated, and 3) determine the extent to which tweets containing fake news expressed polarized political views.\nFor our study, we used the number of retweets to single-out those that went viral within our sample. Tweets within that subset (viral tweets hereafter) are varied and relate to different topics. We consider that a tweet contains fake news if its text falls within any of the following categories described by Rubin et al. BIBREF7 (see next section for the details of such categories): serious fabrication, large-scale hoaxes, jokes taken at face value, slanted reporting of real facts and stories where the truth is contentious. The dataset BIBREF8 , manually labelled by an expert, has been publicly released and is available to researchers and interested parties.\nFrom our results, the following main observations can be made:\nOur findings resonate with similar work done on fake news such as the one from Allcot and Gentzkow BIBREF9 . Therefore, even if our study is a preliminary attempt at characterizing fake news on Twitter using only their meta-data, our results provide external validity to previous research. Moreover, our work not only stresses the importance of using meta-data, but also underscores which parameters may be useful to identify fake news on Twitter.\nThe rest of the paper is organized as follows. "
input_ids = tokenizer.encode(input_text, return_tensors='pt')

print(input_ids)

# Generate outputs with attention scores
outputs = model(input_ids)

# Extract attention scores from the last layer
attentions = outputs.attentions[-1]  # (batch_size, num_heads, seq_len, seq_len)

# Take the average attention scores across all heads
avg_attention = attentions.mean(dim=1).squeeze().detach().numpy()

# Convert token ids back to tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Create a combined heatmap for the attention scores
plt.figure(figsize=(12, 8))
plt.imshow(avg_attention, cmap='viridis', aspect='auto')

# Add labels and titles
plt.colorbar(label='Attention Score')
plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(tokens)), labels=tokens)
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.title("Attention Scores Heatmap")
plt.tight_layout()

# Save the combined heatmap figure
plt.savefig('attention_scores_heatmap.png')

# Optionally, display the combined heatmap
plt.show()

# Find the maximum attention score to set a consistent y-axis scale
max_attention_score = np.max(avg_attention)

# Generate and save individual attention score plots for each token
for i, token in enumerate(tokens):
    attention_scores = avg_attention[i, :]

    # Plotting the attention scores for the current token
    plt.figure(figsize=(14, 6))  # Increase figure width for better visibility
    plt.bar(tokens[:i+1], attention_scores[:i+1], color='skyblue')
    plt.xlabel("Tokens")
    plt.ylabel("Attention Scores")
    plt.title(f"Attention Scores for token '{token}' (position {i+1})")
    plt.xticks(rotation=45, ha='right')  # Rotate and align ticks for better readability
    plt.ylim(0, max_attention_score)  # Set consistent y-axis scale
    plt.tight_layout()

    # Save each individual attention score figure
    plt.savefig(f'attention_scores_token_{i+1}.png')

    # Optionally, display the individual plot
    plt.show()
