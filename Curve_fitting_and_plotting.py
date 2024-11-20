import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Reading the data from the file
with open("MSD_vikki.txt", "r") as fu:
    data = []
    fps = 7.0
    for line in fu:
        ju = line.split()
        data.append([float(ju[0]) / fps, float(ju[1])])

data = np.array(data)

# Extracting time and MSD from the data
time = data[:, 0]
msd = data[:, 1]

# Smooth the MSD data using a moving average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

window_size = 5  # Adjust the window size as needed
msd_smooth = moving_average(msd, window_size)
time_smooth = time[window_size-1:]

# Define the piecewise linear model
def piecewise_linear(t, D1, V1, t0, D2, V2):
    return np.piecewise(t, [t < t0, t >= t0], 
                        [lambda t: 4 * D1 * t + (V1 * t)**2, 
                         lambda t: 4 * D2 * t + (V2 * t)**2])

# Initial guess for the parameters
p0 = [1e-3, 1e-4, time_smooth[len(time_smooth)//2], 1e-3, 1e-4]

# Fit the piecewise linear model to the smoothed data
popt, pcov = curve_fit(piecewise_linear, time_smooth, msd_smooth, p0=p0)

# Extract the fitted parameters
D1, V1, t0, D2, V2 = popt

# Predicted MSD values using the fitted model
msd_pred = piecewise_linear(time, *popt)

# Boltzmann constant
k_B = 1.380649e-23  # J/K
# Temperature in Kelvin
T = 298  # Assuming room temperature

# Viscosity of water at room temperature in Pa.s
eta = 0.0013

# Convert D1 and D2 from μm^2/s to m^2/s
D1_m2_s = D1 * 1e-12
D2_m2_s = D2 * 1e-12

# Calculate hydrodynamic radius (Stokes-Einstein equation)
R_h1 = k_B * T / (6 * np.pi * eta * D1_m2_s) * 1e6  # Convert from meters to micrometers
R_h2 = k_B * T / (6 * np.pi * eta * D2_m2_s) * 1e6  # Convert from meters to micrometers

# Printing the results to the screen
print(f"Diffusion coefficient before transition (D1): {D1:.2e} μm^2/s")
print(f"Velocity before transition (V1): {V1:.2e} μm/s")
print(f"Transition time (t0): {t0:.2e} s")
print(f"Diffusion coefficient after transition (D2): {D2:.2e} μm^2/s")
print(f"Velocity after transition (V2): {V2:.2e} μm/s")
print(f"Hydrodynamic radius before transition (R_h1): {R_h1:.2f} μm")
print(f"Hydrodynamic radius after transition (R_h2): {R_h2:.2f} μm")

# Plotting the results
plt.figure(figsize=(6, 6))

# Log-log scale plot
plt.loglog(time, msd, 'bo', label='Original Data')
plt.loglog(time, msd_pred, 'r-', label='Fitted Curve')
plt.loglog(time_smooth, 4 * D2 * time_smooth, 'c--', label='Diffusive Fit')
plt.xlabel('Time (s)')
plt.ylabel('MSD (μm^2)')
plt.legend(prop={'family': 'Times New Roman'})
plt.title('MSD vs Time with Piecewise Linear Fit (Log-Log Scale)')


plt.tight_layout()

# Save as EPS and PNG files
plt.savefig('msd_plot.eps', format='eps', dpi=300)
plt.savefig('msd_plot.png', format='png', dpi=300)

plt.show()

