import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_node("Team")
G.add_node("Jinghuai Zhang")
G.add_node("Xueqing Wu")
G.add_node("Zhuohao Li")
G.add_node("Ying Li")
G.add_node("Fan Yin")
G.add_node("Kunlin Cai")
G.add_node("Sean Tang")

# Add edges
G.add_edge("Team", "Jinghuai Zhang")
G.add_edge("Team", "Xueqing Wu")
G.add_edge("Team", "Zhuohao Li")
G.add_edge("Team", "Ying Li")
G.add_edge("Team", "Sean Tang")
G.add_edge("Team", "Kunlin Cai")
G.add_edge("Team", "Fan Yin")

# Set up the plot
plt.figure(figsize=(8, 12))
pos = nx.spring_layout(G)

# Draw the graph
nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=10000, arrowsize=20)

# Add labels
nx.draw_networkx_labels(G, pos, {
    "Team": "Team",
    "Jinghuai Zhang": "Jinghuai Zhang\n- Organize team\n- Develop red-teaming algorithms\n- Focus on black-box attacks",
    "Xueqing Wu": "Xueqing Wu\n- Analyze code-based LLMs\n- Perform model training",
    "Zhuohao Li": "Zhuohao Li\n- Examine malicious code\n- Provide technical solutions",
    "Ying Li": "Ying Li\n- Organize team\n- Develop red-teaming algorithms\n- Focus on black-box attacks",
    "Sean Tang": "Seam Tang\n- Analyze code-based LLMs\n- Perform model training",
    "Kunlin Cai": "Kunlin Cai\n- Examine malicious code\n- Provide technical solutions",
    "Fan Yin": "Fan Yin\n- Examine malicious code\n- Provide technical solutions"
}, font_size=16, font_weight="bold")

# Add a title
plt.title("Team Member Responsibilities", fontsize=30, fontweight="bold")

# Remove axis
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.savefig('role.png')
plt.show()