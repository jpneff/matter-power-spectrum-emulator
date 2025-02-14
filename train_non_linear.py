# %%
# Neural network train and test that predicts the non linear power spectrum using the cosmopower code

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display, clear_output

# %%
# Check if GPU is available
device = 'gpu:0' if tf.config.list_physical_devices('GPU') else 'cpu'
print('using', device, 'device \n')

# %%
# Setting the seed for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

# %%
# Training parameters
training_parameters = np.load('./params.npz')
# Training spectrum
training_spectrum = np.load('./spectra.npz')

# %%
import math

# See if there is inf or nan values in the files
def contains_inf_or_nan(arr):
    for value in arr:
        if math.isinf(value) or math.isnan(value):
            return True
    return False

arrays = training_spectrum['features']

for i, arr in enumerate(arrays):
    if contains_inf_or_nan(arr):
        print(f"Array {i} contains inf or nan")

# %%
# Setting the parameters that are being used
model_parameters = ['omega_b', 
                    'omega_c', 
                    'A_s_1e9', 
                    'n_s', 
                    'xi',
                    'h',
                    'z',
                    ]

# %%
# Wavenumber values
k_values = training_spectrum['modes']

# %%
from cosmopower import cosmopower_NN

# Instantiate NN class
cp_nn = cosmopower_NN(parameters=model_parameters, 
                      modes=k_values, 
                      n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )

# %%
# Train
with tf.device(device):
    
    cp_nn.train(training_parameters=training_parameters,
                training_features=training_spectrum['features'],
                filename_saved_model='trained_model',
                validation_split=0.1, # Proportion of the training data to be used for validation. In this case, 10% of the data will be used for validation
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], # Controls how much the model's weights are adjusted with respect to the loss gradient during each update
                batch_sizes=[1024, 1024, 1024, 1024, 1024], # Number of samples that will be passed through the network at once before updating the model's parameters
                gradient_accumulation_steps = [1, 1, 1, 1, 1], # Accumulating gradients over multiple mini-batches before updating the model weights
                patience_values = [200,200,200,200,200], # Prevent overfitting by halting the training process if the model's performance on a validation set does not improve after a specified number of epochs
                max_epochs = [1000,1000,1000,1000,1000], # Total number of complete passes through the entire training dataset
                )
