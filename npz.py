# %%
import numpy as np
import os

# Function to read the file and extract the first 8 parameters
def reading(file_path):
    with open(file_path, 'r') as archive:
        # Read all lines and split into words for processing
        lines = archive.read().splitlines()

    # Extracting values from the first 7 columns of the first non-empty line
    for line in lines:
        if line.strip():  # Ignore empty lines
            values = line.split()[:7]  # Only the first 7 columns
            return [float(val) for val in values]

    return None  # If no valid line is found

if __name__ == "__main__":
    # Pre-allocate lists for storing parameters
    Omega_b, Omega_c, A_s_1e9, n_s, xi, h, z = [], [], [], [], [], [], []

    # Folder where the files are located
    folder_name = "IDE"

    # Loop through the files in the folder
    for i in range(0,10000):
        file_name = f"IDE_{i}.txt"
        file_path = os.path.join(folder_name, file_name)

        # Check if the file exists before reading
        if os.path.exists(file_path):
            params = reading(file_path)
            if params:
                Omega_b.append(params[0])
                Omega_c.append(params[1])
                A_s_1e9.append(params[2])
                n_s.append(params[3])
                xi.append(params[4])
                h.append(params[5])
                z.append(params[6])
        else:
            print(f"File {file_name} doesn't exist.")

    # Convert all lists to numpy arrays directly
    params = {
        'omega_b': np.array(Omega_b),
        'omega_c': np.array(Omega_c),
        'A_s_1e9': np.array(A_s_1e9),
        'n_s': np.array(n_s),
        'xi': np.array(xi),
        'h': np.array(h),
        'z': np.array(z),
    }

    # Save the parameters in a single .npz file
    np.savez('params.npz', **params)

# %%
import numpy as np
import os

# Function to read a specific column from the file (for both k and P_values)
def reading(file_path, column_index):
    column_values = []
    
    with open(file_path, 'r') as archive:
        for line in archive:
            if line.strip():  # Ignore empty lines
                values = line.split()
                column_values.append(float(values[column_index]))

    return column_values

if __name__ == "__main__":
    # Folder where the files are located
    folder_name = "IDE"

    # Read the modes (k-values) from IDE_0.txt, assuming column -4
    file_name = "IDE_0.txt"  
    file_path = os.path.join(folder_name, file_name)
    
    try:
        k = reading(file_path, column_index=-4)  # Reading the k-values from column -4
        k = np.array(k)  # Convert to numpy array

    except FileNotFoundError:
        print(f"File {file_name} doesn't exist.")
        k = None  # Handle case when file is not found

    # Pre-allocate list to store all P_values from multiple files
    P_all = []

    # Loop through all files
    for i in range(10000):
        name = f"IDE_{i}.txt"
        file_path = os.path.join(folder_name, name)

        if os.path.exists(file_path):
            try:
                P_values = reading(file_path, column_index=-3)  # Reading P_values from column -3
                P_all.append(P_values)
            except Exception as e:
                print(f"Fail reading {name}: {e}")
        else:
            print(f"File {file_name} doesn't exist.")

    # Convert list of lists to numpy array
    P_all = np.array(P_all)

    # Save both k-values and P_values in a .npz file
    if k is not None:
        spectra = {
            'modes': k,        # k-values
            'features': P_all  # Non-linear spectra (P_values)
        }

        np.savez('spectra.npz', **spectra)
