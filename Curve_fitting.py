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

# Plotting the results
plt.figure(figsize=(12, 6))



# Printing the results to the screen
print(f"Diffusion coefficient before transition (D1): {D1:.2e} μm^2/s")
print(f"Velocity before transition (V1): {V1:.2e} μm/s")
print(f"Transition time (t0): {t0:.2e} s")
print(f"Diffusion coefficient after transition (D2): {D2:.2e} μm^2/s")
print(f"Velocity after transition (V2): {V2:.2e} μm/s")
print(f"Hydrodynamic radius before transition (R_h1): {R_h1:.2f} μm")
print(f"Hydrodynamic radius after transition (R_h2): {R_h2:.2f} μm")

# Printing the smoothed MSD and time values
print("\nSmoothed MSD and Time values:")
for ts, ms in zip(time_smooth, msd_smooth):
    print(f"Time (s): {ts:.4f}, Smoothed MSD (μm^2): {ms:.4f}")

# Write the fitted curve and smoothed data to a text file
with open('fitted_curve_smoothed_piecewise.txt', 'w') as f:
    f.write("Smoothed and Piecewise Fitted Curve:\n")
    f.write("Time (s)\tFitted MSD (μm^2)\n")
    for t, msd_val in zip(time, msd_pred):
        f.write(f"{t}\t{msd_val}\n")
    
    f.write("\nSmoothed MSD and Time values:\n")
    f.write("Time (s)\tSmoothed MSD (μm^2)\n")
    for ts, ms in zip(time_smooth, msd_smooth):
        f.write(f"{ts}\t{ms}\n")




# Log-log scale plot
plt.subplot(1, 2, 1)
plt.loglog(time, msd, 'bo', label='Original Data')
plt.loglog(time_smooth, msd_smooth, 'go', label='Smoothed Data')
plt.loglog(time, msd_pred, 'r-', label='Fitted Curve')
plt.axvline(x=t0, color='g', linestyle='--', label='Transition Point')
plt.loglog(time_smooth, 4 * D1 * time_smooth, 'm--', label='Diffusive Fit Before Transition')
plt.loglog(time_smooth, 4 * D2 * time_smooth, 'c--', label='Diffusive Fit After Transition')
plt.xlabel('Time (s)')
plt.ylabel('MSD (μm^2)')
plt.legend()
plt.title('MSD vs Time with Piecewise Linear Fit (Log-Log Scale)')

# Normal scale plot
plt.subplot(1, 2, 2)
plt.plot(time, msd, 'bo', label='Original Data')
plt.plot(time_smooth, msd_smooth, 'go', label='Smoothed Data')
plt.plot(time, msd_pred, 'r-', label='Fitted Curve')
plt.axvline(x=t0, color='g', linestyle='--', label='Transition Point')
plt.plot(time_smooth, 4 * D1 * time_smooth, 'm--', label='Diffusive Fit Before Transition')
plt.plot(time_smooth, 4 * D2 * time_smooth, 'c--', label='Diffusive Fit After Transition')
plt.xlabel('Time (s)')
plt.ylabel('MSD (μm^2)')
plt.legend()
plt.title('MSD vs Time with Piecewise Linear Fit (Normal Scale)')

plt.tight_layout()
plt.show()
