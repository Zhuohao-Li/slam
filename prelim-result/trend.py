import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Input and Output data
input_data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
output_data = np.array([5, 12, 19, 29, 40, 48, 59, 71, 81, 90])

# Calculate the percentage
percentage = (output_data / input_data) * 100

# Create a smooth curve
xnew = np.linspace(input_data.min(), input_data.max(), 300)  # 300 points for smoothness
spl = make_interp_spline(input_data, percentage, k=3)  # Cubic spline interpolation
ynew = spl(xnew)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(xnew, ynew, color='#1FD665', linewidth=2)
plt.scatter(input_data, percentage, color='#FF6B6B', zorder=5)  # Original data points
plt.xlabel('Token Budget for Attention Reward')
plt.ylabel('Overlapping ratio (%)')
plt.title('Overlapping Ratio in Neighborhood')
plt.xticks(input_data)
plt.yticks(range(0, 101, 10))
plt.ylim(50, 100)
plt.savefig('trend.png')
plt.show()
