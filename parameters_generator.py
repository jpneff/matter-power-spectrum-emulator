import numpy as np

num_samples = 2500

# Generate random values for each parameter
Omega_b = np.random.uniform(low=0.039, high=0.053, size=(num_samples,))  # Baryonic matter density parameter today
Omega_c = np.random.uniform(low=0.1, high=0.45, size=(num_samples,))     # Cold dark matter density parameter today
A_s_1e9 = np.random.uniform(low=1.61, high=3, size=(num_samples,))    # Primordial scalar power spectrum amplitude, multiplied by 1e9
n_s = np.random.uniform(low=0.9, high=1.1, size=(num_samples,))         # Primordial scalar power spectrum spectral index
xi_ = np.random.uniform(low=-0.5, high=0, size=(num_samples,))           # Dimensionless parameter governing the strength of the DM-DE interaction
h = np.random.uniform(low=0.4, high=0.8, size=(num_samples,))          # Hubble constant in unit of 100 [km/s/Mpc]

# Stack the parameters into a single array
data = np.column_stack((Omega_b, Omega_c, A_s_1e9, n_s, xi_, h))

# Define redshift values
redshifts = np.array([0, 0.5, 1, 1.5])

# Loop through each redshift and save to a file
for z in redshifts:
    # Create a data array for the current redshift
    z_data = np.column_stack((data, np.full((num_samples, 1), z)))
    
    # Define the filename
    filename = f"cosmological_parameters_z{z:.1f}.txt"
    
    # Save the data to the file
    np.savetxt(filename, z_data, fmt='%.6f')

# Optionally, concatenate the files into a single one later
# You can use the following code snippet to do this:
with open('cosmological_parameters.txt', 'w') as outfile:
    for z in redshifts:
        with open(f'cosmological_parameters_z{z:.1f}.txt', 'r') as infile:
            outfile.write(infile.read())

