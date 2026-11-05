# A collection of utilities for ED and Lanczos-type heat capacity calculations.

import numpy as np

# Calculate the FTLM approximation to the n^th power of the Hamiltonian
def _H_n(beta, n, Es, Vs, LVs, sectors = False):

    if not sectors:

        # Expand temperature array to a broadcastable form
        if isinstance(beta, np.ndarray):
            for _ in range(Es.ndim):
                beta = beta[:, np.newaxis]

        Boltzmann = (Es**n) * np.exp(-beta * Es)
        Vs = np.swapaxes(Vs, 1, 2)
        lanczos_eigs = np.matmul(Vs, LVs)
        projs = np.abs(np.matvec(np.conj(lanczos_eigs), LVs[:,0,:]))**2

        ret = np.sum(np.vecdot(Boltzmann, projs), axis = 1)

        return ret
    
    else:

        ret = 0
        # Loop over sectors
        N_sectors = len(Es)
        for i in range(N_sectors):
            d_i = LVs[i].shape[2] # Sector dimension

            # Expand temperature array to a broadcastable form
            if isinstance(beta, np.ndarray):
                Beta = beta.copy()
                for _ in range(Es[i].ndim):
                    Beta = Beta[:, np.newaxis]
            
            Boltzmann = (Es[i]**n) * np.exp(-Beta * Es[i])
            vs = np.swapaxes(Vs[i], 1, 2)
            lanczos_eigs = np.matmul(vs, LVs[i])
            projs = np.abs(np.matvec(np.conj(lanczos_eigs), LVs[i][:,0,:]))**2

            ret += d_i * np.sum(np.vecdot(Boltzmann, projs), axis = 1)

        return ret

# Calculate the FTLM approximation to the n^th power of the Hamiltonian, allowing for a large number of Lanczos runs
def _H_n_large_sampling(beta, n, dims, Es, projs):

    ret = 0
    # Loop over sectors
    N_sectors = dims.shape[0]
    for i in range(N_sectors):

        # Expand temperature array to a broadcastable form
        if isinstance(beta, np.ndarray):
            Beta = beta.copy()
            for _ in range(Es[i].ndim):
                Beta = Beta[:, np.newaxis]

        Boltzmann = (Es[i]**n) * np.exp(-Beta * Es[i])
        ret += dims[i] * np.sum(np.vecdot(Boltzmann, projs[i]), axis = 1)

    return ret

# Calculate the FTLM approximation to the heat capacity, allowing for a large number of Lanczos runs
def _C_ftlm_large_sampling(beta, dims, Es, projs):

    # Calculate partition function
    Z = H_n_large_sampling(beta, 0, dims, Es, projs)

    # Calculate <H> and <H^2>
    H_1 = H_n_large_sampling(beta, 1, dims, Es, projs) / Z
    H_2 = H_n_large_sampling(beta, 2, dims, Es, projs) / Z

    # Calculate heat capacity
    C = (beta**2) * (H_2 - H_1**2)

    return C
    
# Calculate the FTLM approximation to the heat capacity
def _C_ftlm(beta, Es, Vs, LVs, sectors = False):

    # Calculate partition function
    Z = H_n(beta, 0, Es, Vs, LVs, sectors = sectors)

    # Calculate <H> and <H^2>
    H_1 = H_n(beta, 1, Es, Vs, LVs, sectors = sectors) / Z
    H_2 = H_n(beta, 2, Es, Vs, LVs, sectors = sectors) / Z

    # Calculate heat capacity
    C = (beta**2) * (H_2 - H_1**2)

    return C
