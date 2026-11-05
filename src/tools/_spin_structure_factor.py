# A collection of utilities for ED and Lanczos-type spin structure factor calculations.


import numpy as np
import scipy as sp
from quspin.tools.misc import matvec as mv
from quspin.tools.lanczos import lanczos_full
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian


# Compute the second Lanczos basis required for the two-basis LTLM
def second_basis_in_place(beta, H, A, E_1, V_1, LV_1, current_basis, new_basis, dtype = np.complex128):
    """
    Compute the second Lanczos run required for the two-basis implementation of the LTLM using QuSpin, taking the results of the first
    Lanczos run as input.

    Returned is a tuple of NumPy arrays containing, respectively, the Lanczos eigenvalues, eigenvectors, and basis states.

    Parameters
    ----------
    beta : float
        Inverse temperature (in the same units as the energy eigenvalues; see below)
    H : SciPy sparse matrix
        Hamiltonian of interest
    A : list
        Operator list specifying A, in the standard QuSpin operator list format
    E_1 : array-like
        m Lanczos eigenvalues corresponding to the Lanczos run initialized with |r>
    V_1 : array-like
        m Lanczos eigenvectors (stored as the columns, each of length m) corresponding to the Lanczos run initialized with |r>
    LV_1 : array-like
        m Lanczos basis states (stored as the rows) corresponding to the Lanczos run intialized with |r> 
    """
    # Construct initialization state
    evecs = np.matmul(V_1.T, LV_1) # Array of Lanczos eigenvectors (stored in the rows) in the full Hilbert space
    weights = np.exp(-beta * E_1 / 2) * np.matvec(np.conj(evecs), LV_1[0])
    shifted_states = new_basis.Op_shift_sector(current_basis, A, evecs.T, dtype = dtype).T
    Ar = np.dot(weights, shifted_states)
    
    # Compute second Lanczos diagonalization
    E_2, V_2, LV_2 = lanczos_full(H, Ar, evecs.shape[0])

    return (E_2, V_2, LV_2)
