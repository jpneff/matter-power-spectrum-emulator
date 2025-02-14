# %%
# Power spectrum calculator in Interacting Dark Energy Cosmology using the modified pmwd code

import numpy as np
import os

# uncomment to disable NVIDIA GPUs
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
# or pick the device (cpu, gpu, and tpu)
#os.environ['JAX_PLATFORMS'] = 'cpu'

# change JAX GPU memory preallocation fraction
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

# you do not want this
#os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

import jax
jax.print_environment_info()

#!nvidia-smi --query-gpu=gpu_name --format=csv,noheader

import matplotlib.pyplot as plt
    
from pmwd import (
    Configuration,
    Cosmology, SimpleLCDM,
    boltzmann, linear_power, growth,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)

from pmwd.pm_util import fftinv
from pmwd.spec_util import powspec

# %%
import os

def values(Omega_b, Omega_c, A_s_1e9, n_s, xi_, h, z, k, P, k_lin, P_lin, folder_name, file_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Create the full path to the file inside the folder
    full_path = os.path.join(folder_name, file_name)
    
    # Open the file in append mode
    with open(full_path, 'a') as archive:
        # Write the data
        for i in range(len(k)):
            archive.write(f"{Omega_b:.4f} {Omega_c:.4f} {A_s_1e9:.4f} {n_s:.4f} {xi_:.4f} {h:.4f} {z:.4f} {k[i]:.4f} {P[i]:.4f} {k_lin[i]:.4f} {P_lin[i]:.4f}\n")
        archive.write("\n\n")

# Read parameters from the file
parameter_file = 'cosmological_parameters.txt'
parameters = np.loadtxt(parameter_file)

# Ensure the number of parameter sets matches the number of rows in the file
num_parameter_sets = parameters.shape[0]

# Specify the folder to store the output files
output_folder = 'IDE'

# Loop over the parameter sets
for i in range(num_parameter_sets):
    # Extract parameters for the current set
    omega_b = parameters[i, 0]
    omega_c = parameters[i, 1]
    A_s_1e9 = parameters[i, 2]
    n_s = parameters[i, 3]
    xi_ = parameters[i, 4]
    h = parameters[i, 5]
    z = parameters[i, 6]

    # Check if GPU is available
    if jax.default_backend() == 'gpu':
        ptcl_spacing = 1.  # Lagrangian space Cartesian particle grid spacing, in Mpc/h by default
        ptcl_grid_shape = (128,) * 3
    else:
        ptcl_spacing = 4.
        ptcl_grid_shape = (64,) * 3

    # Configuration stores static configuration and parameters for which we do not need derivatives
    conf = Configuration(ptcl_spacing, ptcl_grid_shape, a_stop=1/(1+z), mesh_shape=2) # 2x mesh shape

    # Initialize cosmology with parameters from the file
    # Cosmology stores interesting parameters, whose derivatives we need.
    cosmo = Cosmology(conf, A_s_1e9=A_s_1e9, n_s=n_s, Omega_c=omega_c, Omega_b=omega_b, xi_=xi_, h=h)
    cosmo = boltzmann(cosmo, conf)

    # Generate a white noise field, and scale it with the linear power spectrum
    seed = 0
    modes = white_noise(seed, conf)
    modes = linear_modes(modes, cosmo, conf)

    # Solve LPT at some early time
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl.disp.std(), ptcl.vel.std()

    # N-body time integration from the LPT initial conditions
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    ptcl.disp.std(), ptcl.vel.std()

    # Scatter the particles to mesh to get the density field
    dens = scatter(ptcl, conf)

    # Measure and plot the matter density power spectra
    k, P, _, _ = powspec(dens, conf.cell_size)

    dens_lin = fftinv(modes, shape=conf.ptcl_grid_shape, norm=conf.ptcl_spacing)
    dens_lin *= growth(conf.a_stop, cosmo, conf)
    k_lin, P_lin, _, _ = powspec(dens_lin, conf.ptcl_spacing)
     
    # Save the data to a text file in the specified folder
    values(omega_b, omega_c, A_s_1e9, n_s, xi_, h, z, k, P, k_lin, P_lin, output_folder, f'IDE_{i}.txt')
