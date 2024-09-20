import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
token_budgets = [256, 1024]
full = [1, 1]    # Example values
slam = [0.89, 0.94]   # Example values
quest = [0.85, 0.98]   # Example values
h2o = [0.35, 0.60]     # Example values

# Bar width and positions
bar_width = 0.20
positions = np.arange(len(token_budgets))

# Plotting
fig, ax = plt.subplots()

bars1 = ax.bar(positions - 1.5 * bar_width, full, bar_width, label='Full', color='#FFA07A')
bars2 = ax.bar(positions - 0.5 *bar_width, slam, bar_width, label='Slam', color='#E49BBB')
bars2 = ax.bar(positions + 0.5 *bar_width, quest, bar_width, label='Quest', color='#87CEEB')
bars3 = ax.bar(positions + 1.5*bar_width, h2o, bar_width, label='H2O', color='#4682B4')

# Labels and Title
ax.set_xlabel('Token Budget')
ax.set_ylabel('Top-10 Recall Rate')
ax.set_title('Top-10 Recall Rate by Token Budget')
ax.set_xticks(positions)
ax.set_xticklabels(token_budgets)
ax.set_ylim(0, 1)  # Set the y-axis range from 0 to 1
ax.legend()

# Show the plot
plt.tight_layout()
plt.savefig('token.png')
plt.show()
