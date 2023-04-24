import numpy as np
import matplotlib.pyplot as plt


def c_v(v, c0, v0, m):
    """Returns the capacitance (in picoFarad) as function of the voltage (in volts)"""
    return c0 / ((1 + v / v0) ** m)


def v_c(c, c0, v0, m):
    """Returns the voltage (in volts) as function of the capacitance (in picoFarad)"""
    return v0 * (((c0 / c) ** (1 / m)) - 1)


# Set the parameters
# c0 = 867.3
c0 = 10  # in picoFarad
v0 = 2.9  # in volts
m = 1.66

# Create an array of v values
v = np.arange(0, 25, 0.01)
c = np.arange(0.25, 10, 0.01)

# Calculate the corresponding C values
C = c_v(v, c0, v0, m)

# Calculate the corresponding V values
V = v_c(c, c0, v0, m)

print("min capacitance:", min(C))
print("max capacitance:", max(C))

# Plot the function
plt.plot(v, C, 'b-', label="C(v)")  # Polt C(v)
plt.plot(V, c, 'r-', label="V(c)")  # Polt V(c) (inverse: V on x-axis and c on y-axis)

# Add axis labels and a title
plt.legend()
plt.xlabel('v (V)')
plt.ylabel('C (pF)')
plt.title('C(v)')

# Show the plot
plt.show()
