import numpy as np

# Constants
a_0 = 0.0529e-9  # First Bohr radius

# Wave function
def wave_func(r):
    normalization_factor = 1 / (4 * np.sqrt(2 * np.pi * a_0**3))
    psi_value = normalization_factor * (2 - r / a_0) * np.exp(-r / (2 * a_0))
    return np.float64(psi_value)

# Probability function
def prob_density(r, n):
    psi_val = wave_func(r)
    return np.abs(psi_val)**2 * r**n * r**2 # The function on the excercise sheet was wrong. There must be another r^2

# Newton Cortes function
def newton_cotes(x, f_vals):
    
    h = (x[-1] - x[0]) / (len(x) - 1) #Calculate step size
    integral_sum = 0.0

    
    for i in range(0, len(x) - 2, 2):  # calculate the integral
        integral_sum += (f_vals[i] + 4 * f_vals[i + 1] + f_vals[i + 2])

    return (h / 3) * integral_sum



interval_a = 1e-10  # Avoid singularity at r=0
upper_bounds = [10 * a_0, 15 * a_0, 40 * a_0, 60*a_0,1000*a_0]  # Define upper bounds, to get the value for b -> infinite


results = []
for t in upper_bounds:
    
    n_points = 10000  # Number of points
    r_values = np.linspace(interval_a, t, n_points)
    
    integrand_values = prob_density(r_values, 1)  # Calculate mean radius
    integrand_values_2 = prob_density(r_values, 2)  # Calculate standard deviation


    integral_result = 4 * np.pi * newton_cotes(r_values, integrand_values)
    integral_result_2 = 4 * np.pi * newton_cotes(r_values, integrand_values_2)
    devitation = np.sqrt(np.sqrt(integral_result_2))
    results.append((t, integral_result, devitation))
    print(f"Upper bound: {t/a_0} * a_0, Integral value: {integral_result}, Devitation: {devitation}")


