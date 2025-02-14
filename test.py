# %%
# Test emulator

# %%
from cosmopower import cosmopower_NN
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# %%
# Function that calls the trained neural network and calculates the non linear power spectrum based on the desired parameters
def non_linear_spectrum(omega_b, omega_c, A_s_1e9, n_s, xi, h, z):
    cp_nn = cosmopower_NN(restore=True, 
                      restore_filename='trained_model',
                      )
    
    parameters = {'omega_b': [omega_b], 
                  'omega_c': [omega_c], 
                  'A_s_1e9': [A_s_1e9], 
                  'n_s': [n_s], 
                  'xi': [xi],
                  'h': [h],
                  'z': [z],
                 }

    power_spectra = cp_nn.predictions_np(parameters)[0]
    power_spectra = gaussian_filter(power_spectra, sigma=0)

    k_modes = cp_nn.modes

    return power_spectra, k_modes

power_spectra, k_modes = non_linear_spectrum(0.05, 0.3, 2, 0.96, -0.1, 0.7, 0)

# %%
#Calculate the original spectrum for the desired parameters just for comparison, not necessary
from pmwd import (
    Configuration,
    Cosmology, SimpleLCDM,
    boltzmann, linear_power, growth,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.spec_util import powspec
from pmwd.pm_util import fftinv
from pmwd.vis_util import simshow

ptcl_spacing = 1           
ptcl_grid_shape = (256,) * 3  
conf = Configuration(ptcl_spacing, ptcl_grid_shape, a_stop=1/(1+0), mesh_shape=2)  

cosmo = Cosmology(conf, A_s_1e9=2, n_s=0.96, Omega_c=0.3, Omega_b=0.05, xi_=-0.1, h=0.7)
cosmo = boltzmann(cosmo, conf)

seed = 0
modes = white_noise(seed, conf)
modes = linear_modes(modes, cosmo, conf)
    
ptcl, obsvbl = lpt(modes, cosmo, conf)
ptcl.disp.std(), ptcl.vel.std()
    
ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
ptcl.disp.std(), ptcl.vel.std()

dens = scatter(ptcl, conf)
k, P, _, _ = powspec(dens, conf.cell_size)

dens_lin = fftinv(modes, shape=conf.ptcl_grid_shape, norm=conf.ptcl_spacing)
dens_lin *= growth(conf.a_stop, cosmo, conf)
k_lin, P_lin, _, _ = powspec(dens_lin, conf.ptcl_spacing)

# %%
# CLASS activation for comparison, not necessary

from classy import Class

cosmo = Class()

params = {'output': 'tCl mPk',
          'z_max_pk': 5,
          'P_k_max_1/Mpc': 10.,
          'nonlinear_min_k_max': 100.,
          'N_ncdm' : 1,
          'N_eff' : 3.046,
          'omega_b': 0.05,
          'omega_cdm': 0.3,
          'h': 0.7,
          'n_s': 0.96,
          'ln10^{10}A_s': 2,
          'YHe': 0.4,
          }

cosmo.set(params)
cosmo.compute()

z = 0

spectrum_class = np.array([cosmo.pk(ki, z) for ki in k_modes])

# %%
# Graphic comparison
fig = plt.figure(figsize=(20,10))
plt.loglog(k, P, 'orange', label = 'pmwd Non-Linear', linewidth=3)
plt.loglog(k_modes, power_spectra, 'blue', label = 'CosmoPower Non-Linear', linestyle =  '--',linewidth=3)
plt.loglog(k_modes, spectrum_class, 'red', label = 'CLASS', linewidth=3)
plt.xlabel('$k$ [Mpc$^{-1}]$', fontsize=20)
plt.ylabel('$P(k)[\mathrm{Mpc}^3]$', fontsize=20)
plt.legend(fontsize=20)
plt.show()

import glob

files = glob.glob('wCDM_*.txt')

fig = plt.figure(figsize=(20,10))

plt.axhline(y=1, color='black', linestyle='-', linewidth=2)

for file in files:
    data = np.loadtxt(file)
    
    k = data[:, 7]    
    P = data[:, 8]  
    P_lin = data[:, 10]
   
    plt.loglog(k, P/P_lin, alpha=0.5)

plt.xlabel(r'$k [Mpc^{-1}]$', fontsize=20)
plt.ylabel(r'$P_{\text{NL}}/P_{\text{LIN}}$', fontsize=20)
plt.xlim(0.1, 3.5164)
plt.grid(True, which="both", ls="--")
plt.show()
#plt.savefig('pmwd_wCDM_1000_ratio.png')
