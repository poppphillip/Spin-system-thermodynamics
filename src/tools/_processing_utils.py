# A collection of utilities for processing simulated data.


# Lorentzian of unit height
def Lorentz(x, center, epsilon):
    ret = (1 / np.pi) * (epsilon / ((x - center)**2 + epsilon**2))

    return ret

# Gaussian of unit height
def Gauss(x, center, sigma):
    ret = np.exp(-((x - center)**2) / sigma**2)

    return ret

# Lorentzian broadening function
def LorentzBroad(x, centers, heights, epsilon):
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
def GaussBroad(x, centers, heights, sigma):
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
