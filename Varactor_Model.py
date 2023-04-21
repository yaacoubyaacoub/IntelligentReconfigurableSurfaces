import numpy as np
import matplotlib.pyplot as plt


# Define the function
def c(v, c0, v0, m):
    return c0 / ((1 + v / v0) ** m)


# Set the parameters
c0 = 867.3
v0 = 2.9
m = 1.66

# Create an array of v values
v = np.arange(0, 25, 0.01)


# Calculate the corresponding C values
C = c(v, c0, v0, m)

print("min capacitance:", min(C))
print("max capacitance:", max(C))

# Plot the function
plt.plot(v, C, 'b-')

# Add axis labels and a title
plt.xlabel('v (V)')
plt.ylabel('C (pF)')
plt.title('C(v)')

# Show the plot
plt.show()
