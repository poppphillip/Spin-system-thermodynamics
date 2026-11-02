# This file contains utilities to construct various 2D lattices and identify their bonds


import numpy as np
import time
from itertools import product


# Construct a 2D rotation matrix
def R(theta):
    """
    Construct a 2D rotation matrix which rotates a vector counterclockwise about the z-axis by a specified angle.
    Returned is a NumPy array.

    Parameters
    ----------
    theta : float
        Rotation angle in degrees
    """
    theta_rad = np.deg2rad(theta) # Rotation angle in radians
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])

    return R

# Translate a 2D vector assuming periodic boundary conditions
def add2D_pbc(a, b, c_1, c_2):
    """
    Perform addition of 2D vectors assuming periodic boundary conditions.

    Returned is the modular vector sum a + b.

    Parameters
    ----------
    a : array-like
        First vector to add
    b : array-like
        Second vector to add
    c_1 : array-like
        First lattice vector in Cartesian coordinates
    c_2 : array-like
        Second lattice vector in Cartesian coordinates
    """
    # Construct matrix which converts Cartesian coordinates to the desired lattice coordinates
    cart_to_lat = (1 / (c_1[0] * c_2[1] - c_2[0] * c_1[1])) * np.array([[c_2[1], -c_2[0]], [-c_1[1], c_1[0]]])
    
    # Calculate lattice coordinates of the normal (i.e. non-modular) vector sum
    lat = np.matmul(cart_to_lat, (a + b))

    # Find equivalent lattice coordinates in the first unit cell
    first_uc_lat = np.array([np.sign(lat[0]) * (lat[0] % 1), np.sign(lat[1]) * (lat[1] % 1)])

    # Calculate Cartesian coordinates of the first unit cell-positions
    first_uc_cart = first_uc_lat[0] * c_1 + first_uc_lat[1] * c_2

    return first_uc_cart

# Translate a 3D vector assuming periodic boundary conditions
def add3D_pbc(a, b, c_1, c_2, c_3):
    """
    Perform addition of 3D vectors assuming periodic boundary conditions.

    Returned is the modular vector sum a + b.

    Parameters
    ----------
    a : array-like
        First vector to add
    b : array-like
        Second vector to add
    c_1 : array-like
        First lattice vector in Cartesian coordinates
    c_2 : array-like
        Second lattice vector in Cartesian coordinates
    c_3 : array-like
        Third lattice vector in Cartesian coordinates
    """
    # Construct matrix which converts Cartesian coordinates to the desired lattice coordinates
    cart_to_lat = np.linalg.inv(np.array([c_1, c_2, c_3]).T)
    
    # Calculate lattice coordinates of the normal (i.e. non-modular) vector sum
    lat = np.matmul(cart_to_lat, (a + b))

    # Find equivalent lattice coordinates in the first unit cell
    first_uc_lat = np.array([np.sign(lat[0]) * (lat[0] % 1), np.sign(lat[1]) * (lat[1] % 1), np.sign(lat[2]) * (lat[2] % 1)])

    # Calculate Cartesian coordinates of the first unit cell-positions
    first_uc_cart = first_uc_lat[0] * c_1 + first_uc_lat[1] * c_2 + first_uc_lat[2] * c_3

    return first_uc_cart

# Build a kagome cluster of given dimensions and find its nearest-neighbor pairs
def kagome_cluster(n_rows, n_cols, side_length = 1., sublattices = False):
    """
    Generate the Cartesian-coordinate site positions of a kagome cluster of given dimensions, and find the corresponding nearest-neighbor site pairs assuming periodic boundary
    conditions.
    
    Returned is a list containing four NumPy arrays, the first of which contains the site positions as Cartesian ordered pairs, the second of which contains the nearest-neighbor
    site pairs, and the third and fourth of which contain the superlattice vectors.
    
    Parameters
    ----------
    n_rows : int
        Number of rows of kagome unit cells
    n_cols : int
        Number of columns of kagome unit cells
    side_length : float
        Optional, side length of the triangular unit cell in arbitrary units; default value is 1.
    """
    # Hexagonal Bravais lattice vectors and superlattice vectors (where the triangle side length is set to unity)
    a = 2 * side_length
    a_1 = a * np.array([1., 0.])
    a_2 = (a / 2) * np.array([1., np.sqrt(3)])
    v_1 = n_cols * a_1
    v_2 = n_rows * a_2
    # Construct kagome basis vectors
    h = side_length / np.sqrt(3) # Magnitude of the basis vectors
    r_0 = np.matmul(R(270), np.array([h, 0]))
    r_1 = np.matmul(R(150), np.array([h, 0]))
    r_2 = np.matmul(R(30), np.array([h, 0]))
    
    # Generate sites
    sites = []
    if sublattices:
        sl_dict = {"1" : [], "2" : [], "3" : []}
    for i in range(n_rows):
        for j in range(n_cols):
            sites.append(add2D_pbc((r_0 + (j + 1) * a_1 + (i + 1) * a_2), np.array([0., 0.]), v_1, v_2))
            sites.append(add2D_pbc((r_1 + (j + 1) * a_1 + (i + 1) * a_2), np.array([0., 0.]), v_1, v_2))
            sites.append(add2D_pbc((r_2 + (j + 1) * a_1 + (i + 1) * a_2), np.array([0., 0.]), v_1, v_2))
            if sublattices:
                sl_dict["1"].append(((len(sites) - 3), add2D_pbc(((j + 1) * a_1 + (i + 1) * a_2), np.array([0., 0.]), v_1, v_2)))
                sl_dict["2"].append(((len(sites) - 2), add2D_pbc(((j + 1) * a_1 + (i + 1) * a_2), np.array([0., 0.]), v_1, v_2)))
                sl_dict["3"].append(((len(sites) - 1), add2D_pbc(((j + 1) * a_1 + (i + 1) * a_2), np.array([0., 0.]), v_1, v_2)))
    
    # Find nearest-neighbor site pairs
    pairs = []
    for i in range(len(sites)):
        for j in range(len(sites)):
            if np.allclose(np.dot(sites[i] - sites[j], sites[i] - sites[j]), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
            elif np.allclose(np.dot(sites[i] - sites[j] + v_1, sites[i] - sites[j] + v_1), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
            elif np.allclose(np.dot(sites[i] - sites[j] - v_1, sites[i] - sites[j] - v_1), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
            elif np.allclose(np.dot(sites[i] - sites[j] + v_2, sites[i] - sites[j] + v_2), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
            elif np.allclose(np.dot(sites[i] - sites[j] - v_2, sites[i] - sites[j] - v_2), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
            elif np.allclose(np.dot(sites[i] - sites[j] + v_1 + v_2, sites[i] - sites[j] + v_1 + v_2), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
            elif np.allclose(np.dot(sites[i] - sites[j] + v_1 - v_2, sites[i] - sites[j] + v_1 - v_2), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
            elif np.allclose(np.dot(sites[i] - sites[j] - v_1 - v_2, sites[i] - sites[j] - v_1 - v_2), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
            elif np.allclose(np.dot(sites[i] - sites[j] - v_1 + v_2, sites[i] - sites[j] - v_1 + v_2), 1.) and [j, i].copy() not in pairs:
                pairs.append([i, j].copy())
    
    if sublattices:
        return [np.array(sites), np.array(pairs), v_1, v_2, sl_dict]
    
    else:
        return [np.array(sites), np.array(pairs), v_1, v_2]

# Build a kagome cluster whose number of sites is a multiple of 9 and find its nearest-neighbor pairs
def kagome_9(n_rows, n_cols, side_length = 1.):
    """
    Generate the Cartesian-coordinate site positions of a kagome cluster of given dimensions whose
    total number of sites is a multiple of 9, and find the corresponding nearest-neighbor site pairs assuming 
    periodic boundary conditions.
    
    Returned is a list containing four NumPy arrays, the first of which contains the site positions as Cartesian ordered pairs, the second of which contains the nearest-neighbor
    site pairs, and the third and fourth of which contain the superlattice vectors.
    
    Parameters
    ----------
    n_rows : int
        Number of rows of 9-site kagome unit cells
    n_cols : int
        Number of columns of 9-site kagome unit cells
    side_length : float
        Optional, side length of the triangular unit cell in arbitrary units; default value is 1.
    """
    # Hexagonal Bravais lattice vectors and superlattice vectors
    a_1 = 2 * side_length

# Find the reciprocal lattice vectors of a given 2D lattice
def rlv_2D(a_1, a_2):
    """
    Find the reciprocal lattice vectors corresponding to a given set of 2D lattice vectors.

    Returned is a list [b_1, b_2] of the reciprocal lattice vectors, which have type numpy.ndarray.

    Parameters
    ----------
    a_1 : array-like
        First lattice vector
    a_2 : array-like
        Second lattice vector
    """
    # Construct b_1
    b_1 = (2 * np.pi / np.dot(a_1, (R(90) @ a_2))) * R(90) @ a_2
    # Construct b_2
    b_2 = (2 * np.pi / np.dot(a_2, (R(90) @ a_1))) * R(90) @ a_1

    return [np.asarray(b_1), np.asarray(b_2)]

# Find the numerically allowed wavevectors of a given supercell
def num_q(n_1, n_2, b_1, b_2):
    """
    Find the numerically allowed wavevectors in the first Brillouin zone of the primitive cell of the kagome lattice corresponding to a
    supercell of given dimensions.

    Returned is a NumPy array of the allowed wavevectors [q_1, q_2,...] in Cartesian coordinates of the reciprocal lattice, which have type
    numpy.1darray.

    Parameters
    ----------
    n_1 : int
        Number of unit cells in the supercell along the lattice direction a_1
    n_2 : int
        Number of unit cells in the supercell along the lattice direction a_2
    b_1 : array-like
        First reciprocal lattice vector
    b_2 : array-like
        Second reciprocal lattice vector 
    """
    # Construct primitive vectors of the first Brillouin zone
    p_1 = (1 / np.sqrt(3)) * R(-30) @ b_2
    p_2 = (1 / np.sqrt(3)) * R(30) @ b_1

    # Initialize return list
    q_list = []

    # Generate allowed wavevectors
    for i in range(n_1):
        for j in range(n_2):
            q = add2D_pbc(np.array([0., 0.]), np.asarray((i / n_1) * b_1 + (j / n_2) * b_2), p_1, p_2)
            # q = np.asarray((i / n_1) * b_1 + (j / n_2) * b_2)
            q_list.append(q)

    return np.array(q_list)

# Check that a given translation is a symmetry of a given cluster assuming periodic boundary conditions, and return permuted site indices
def istranssym_pbc(sites, c_1, c_2, vector, round_to = 6):
    """
    Determine whether a given uniform translation is a symmetry of a given cluster assuming periodic boundary conditions; and if it is,
    return the permutation of site indices defined by the translation. That is, if the translation T is indeed a symmetry of the cluster, then
    each site i of the original cluster is mapped bijectively to some other site j, T(i) = j. This mapping defines the permutation
    {T(0),...,T(N - 1)} of the N site indices {0,...,(N - 1)} of the untranslated cluster.

    If the given translation is indeed a symmetry, returned is a NumPy array of the permuted site indices as defined above.

    Parameters
    ----------
    sites : array-like
        Cartesian ordered pair site coordinates of the untranslated cluster
    c_1 : array-like
        First lattice vector in Cartesian coordinates
    c_2 : array-like
        Second lattice vector in Cartesian coordinates
    vector : array-like
        Cartesian vector to test as a symmetry of the input cluster
    round_to : int
        Optional, number of decimal places to which to round all site coordinates before checking their equivalence; default value is 6
    """
    # Create a dictionary whose keys are the untranslated site coordinates and whose values are the untranslated site indices
    site_dict = {tuple(np.round(sites[i], round_to)) : i for i in range(len(sites))}

    # Create list of translated site coordinates
    trans_sites = [tuple(np.round(add2D_pbc(site, vector, c_1, c_2), round_to)) for site in sites]

    # Check whether translated sites correspond to an untranslated site
    permutation = np.zeros(len(sites), dtype = int)
    for i in range(len(trans_sites)):
        if trans_sites[i] in site_dict.keys():
            # permutation[i] = site_dict[trans_sites[i]]
            permutation[site_dict[trans_sites[i]]] = i

    # Check whether the given translation is actually a symmetry of the given cluster
    for val in site_dict.values():
        if val not in permutation:
            raise Exception("This cluster is not translation-invariant with respect to this vector")

    return permutation

# Create a basic checkerboard lattice
def checkerboard(n_rows, n_cols):

    # Initialize primitive lattice vectors and site positions in the primitive unit cell
    a_1 = np.array([1., -1.])
    a_2 = np.array([1., 1.])
    site_1 = np.zeros(2, dtype = np.float64)
    site_2 = np.array([1., 0.])

    # Create superlattice vectors
    v_1 = n_cols * a_1
    v_2 = n_rows * a_2

    # Generate sites

    sites = []

    for j in range(0, n_cols):
        for i in range(0, n_rows):
            sites.append(site_1 + j * a_1 + i * a_2)
            sites.append(site_2 + j * a_1 + i * a_2)

    # Identify short bonds

    short_bonds = []

    for i in range(len(sites)):
        for j in range(i):
            for k in range(-1, 2):
                for l in range(-1, 2):
                    displacement = sites[i] - sites[j] + k * v_1 + l * v_2
                    if np.allclose(np.linalg.norm(displacement), 1.):
                        short_bonds.append([i, j])

    # Identify long bonds

    long_bonds = []
    evens = [i for i in range(0, len(sites), 2)]
    odds = [i for i in range(1, len(sites), 2)]

    for i in range(1, len(evens)):
        for j in range(i):
            for k in range(2):
                displacement = sites[evens[i]] - sites[evens[j]] - k * v_1
                if (np.allclose(displacement, a_1, atol = 1E-6) | np.allclose(displacement, -a_1, atol = 1E-6)):
                    long_bonds.append([evens[i], evens[j]])

    for i in range(1, len(odds)):
        for j in range(i):
            for k in range(2):
                displacement = sites[odds[i]] - sites[odds[j]] - k * v_2
                if (np.allclose(displacement, a_2, atol = 1E-6) | np.allclose(displacement, -a_2, atol = 1E-6)):
                    long_bonds.append([odds[i], odds[j]])

    return (np.array(sites), short_bonds, long_bonds, v_1, v_2)
