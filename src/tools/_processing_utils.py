# A collection of utilities for processing simulated data.


import numpy as np


# Find the indices at which an array is closest to a given set of values
def _find_indices(x, values):

    indices = [np.argmin(np.abs(x - value)) for value in values]

    return tuple(indices)

# Convert a dictionary whose keys are (real) values of an independent variable and whose values are the corresponding values of some function of
# this variable to a 2D NumPy array, wherein the independent variables are sorted in increasing order and the dependent variables are sorted concomitantly
def _dict_sort(dictionary, select = False, min_val = 0., max_val = 0.):

    # Find proper index order for the sorted arrays
    sorted_inds = np.argsort(np.array(list(dictionary.keys())))

    # Convert keys and values into sorted arrays
    keys_arr = np.array(list(dictionary.keys()))[sorted_inds]
    values_arr = np.array(list(dictionary.values()))[sorted_inds]

    if not select:
        return np.array([keys_arr, values_arr])
    
    else:
        min_ind, max_ind = find_indices(keys_arr, [min_val, max_val])
        return np.array([keys_arr[min_ind:max_ind], values_arr[min_ind:max_ind]])

# Perform averaging of dynamical spin structure factor over Lanczos runs within a given magnetization transition channel
def _dssf_lanczos_ave(samples, channel, omega, epsilon):

    # Extract ranges of the Lanczos data to be broadened
    cut_samples = [dict_sort(sample[channel], select = True, min_val = min(omega), max_val = max(omega)) for sample in samples]

    # Broaden and average Lanczos data
    broadened_samples = np.array([lz.LorentzBroad(omega, cut_sample[0], cut_sample[1], epsilon) for cut_sample in cut_samples])
    average = np.sum(broadened_samples, axis = 0)
    Z = np.real(np.sum(np.array([sample["Z"] for sample in samples]))) # Calculate partition function
    average /= Z

# Lorentzian of unit height
def _Lorentz(x, center, epsilon):
    ret = (1 / np.pi) * (epsilon / ((x - center)**2 + epsilon**2))

    return ret

# Gaussian of unit height
def _Gauss(x, center, sigma):
    ret = np.exp(-((x - center)**2) / sigma**2)

    return ret

# Lorentzian broadening function
def _LorentzBroad(x, centers, heights, epsilon):
    """
    Applies a Lorentzian broadening to a given data set.
    
    Returned is the broadened data set as a NumPy array.
    
    Parameters
    ----------
    x : array-like
        x-axis of data set to be broadened
    centers : array-like
        Centers of the Lorentzian distributions to be used for the broadening
    heights : array-like
        Heights of the Lorentzian distributions centered at "centers" (see above); should have the same dimension as "centers"
    epsilon : float
        Smearing parameter
    """
    # Insert dummy dimensions to enable broadcasting
    C = centers[:, np.newaxis]
    X = x[np.newaxis, :]

    # Compute broadened data set
    arr = np.multiply(Lorentz(X, C, epsilon), heights[:, np.newaxis])
    broadened = np.sum(arr, axis = 0)

    return broadened

# Gaussian broadening function
def _GaussBroad(x, centers, heights, sigma):
    """
    Applies a Gaussian broadening to a given data set.
    
    Returned is the broadened data set as a NumPy array.
    
    Parameters
    ----------
    x : array-like
        x-axis of data set to be broadened
    centers : array-like
        Centers of the Gaussian distributions to be used for the broadening
    heights : array-like
        Heights of the Gaussian distributions centered at "centers" (see above); should have the same dimension as "centers"
    sigma : float
        Standard deviation of the Gaussian distributions centered at "centers" (see above)
    """
    # Insert dummy dimensions to enable broadcasting
    C = centers[:, np.newaxis]
    X = x[np.newaxis, :]

    # Compute broadened data set
    arr = np.multiply(Gauss(X, C, sigma), heights[:, np.newaxis])
    broadened = np.sum(arr, axis = 0)

    return broadened
