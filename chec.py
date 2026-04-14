import matplotlib.pyplot as plt

# 1. Setup the Data
labels = ['Random Forest', 'Gradient Boosting', 'XGBoost']
accuracies = [98.33, 98.11, 98.33]
# Using the hex codes that match your original image's theme
fill_colors = ['#9A9BA4', '#FCD071', '#75EBF7'] 
edge_colors = ['#5B5C65', '#EBA421', '#17CBE1']

# 2. Configure the Figure layout 
# A smaller height (3) naturally brings the horizontal rows closer together
fig, ax = plt.subplots(figsize=(10, 3))

# 3. Create the Horizontal Bars
# 'height' parameter controls how thin the bars are (lower number = thinner bars)
bars = ax.barh(labels, accuracies, color=fill_colors, edgecolor=edge_colors, height=0.3, linewidth=2)

# Ensure 'Random Forest' appears at the top
ax.invert_yaxis()

# 4. Add the Baseline (Random 25.00%)
ax.axvline(x=25, color='#CD6D64', linestyle='--', linewidth=1.2, zorder=0)
ax.text(26, 2.4, 'Random 25.00%', color='#CD6D64', va='center', fontsize=9)

# 5. Add Text Labels inside/near the bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    # Placing the text inside the right edge of the bar
    ax.text(width - 2, bar.get_y() + bar.get_height()/2, 
            f'{width}%', 
            ha='right', va='center', fontweight='bold', fontsize=12, color='black')

# 6. Final Polish & Formatting
ax.set_xlim(0, 110)
ax.set_xlabel('Accuracy (%)')
ax.set_title('A1 Final Accuracy (After Interactive Diagnosis) - Post-questioning performance', pad=15)

# Clean up the borders (spines) to match your clean aesthetic
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#DDDDDD')
ax.spines['bottom'].set_color('#DDDDDD')

# Add subtle vertical gridlines for readability
ax.xaxis.grid(True, linestyle='-', color='#EEEEEE', zorder=0)
ax.set_axisbelow(True) # Puts gridlines behind the bars

plt.tight_layout()
plt.show()