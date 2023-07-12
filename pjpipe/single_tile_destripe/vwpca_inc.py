# -*- coding: utf-8 -*-
"""
Copyright (C) 2007 Tamas Budavari and Vivienne Wild (MAGPop)

This script contains the functions needed to perform the robust
itrative PCA technique in Budavari et al. 2009 (MNRAS 394, 1496–1502)
Each method has been written out separately and the method can be run
using the wrapper function. There are two wrapper functions depending
on if the data entered has bad values that should not be used, or whether
the data fine. If there are bad values, these are indicated in an error
array, where bad values should be set to equal zero at the locations
where the data is bad. Bad values are replaced with values reconstructed
from the current eigen basis.
Wrapers perform a single iteration of the robust PCA which updates
the current eigen basis with a new eigen basis using the new, given
data.

Defining some parameters

    'Scale' is the statistical name given to something
         that estimates the spread of the data (i.e., such as a
         standard deviation, or MAD)

    'Delta' is the breakdown point (between 0 to 0.5). This controls the
        robustness. The lower this value, the more sceptible the
        method is to outlier contamination. Increasing delta improves
        the robustness but at the cost of speed. Budavari et al 2009 use 0.5.

    'Observation vector' is the statistical name for data that has been centred
         by subtracting the mean.

    'SVD' or singular value decomposistion factorises a Matrix, A into 3 new
        matrices, U, W, V^T. This factorisation happens to be related to the
        eigensystem of the origional matrix, where U are the eigenvectors
        of A.A^T, and V are the eigenvectors of A^T.A. W are the sigular
        values, which is a diagonal matrix containing the positive
        roots of the eigenvalues of both U and V,


Notes:
    In Budavari et al 2009 section 3.2, they say that they use a c=0.787 for a
    breakdown point (delta)=0.5, and to match a normal distribution but in the
    VW IDL implimentation, this was not implimented, which escentially  means
    that c=1. I have added c^2 as an optional parameter throughout the code if
    one wants to change c^2 to something else.
    TODO: Make code more general for different robust functions that might
        take extra parameters. This can be changed by adding *args where needed
"""
import numpy as np
from . import vwpca_normgappy as gappy


def cauchy_like_function(t, c_sq=1):
    """
    Cauchy functions are bounded functions. Funtion defined in Budavari
    et al. 2009 in eq. (23).

    Parameters
    ----------
    t : float or array_like
       Dependent variable. Here t = residuals^2/sigma^2.
    c_sq : float, optional
        The parameter that sets how when outliers are down weighted
        (the strictness). The default is 1.

    Returns
    -------
    rho : float or array_like
        The robust function used in Budavari et al 2009, eq. (23).

    """
    half_pi = np.pi / 2.0
    rho = np.arctan(half_pi * t / c_sq) / half_pi

    return rho


def derivate_of_cauchy_like_function(t, c_sq=1):
    """
    Evaluates t using the derivative of the cauchy-like function.

    Parameters
    ----------
    t : float or array_like
       Dependent variable. Here t = residuals^2/sigma^2.
    c_sq : float, optional
        The parameter that sets how when outliers are down weighted
        (the strictness). The default is 1.

    Returns
    -------
    derivative : float or array_like
        The derivative of the robust function used in Budavari
        et al 2009, eq. (23).

    """
    half_pi = np.pi / 2.0
    derivative = c_sq / (1.0 + (half_pi * t / c_sq) ** 2)

    return derivative


def weight_star(t, c_sq=1, robust_function=cauchy_like_function):
    """
    W* = rho(t)/t, where rho is a robust function.
    (Defined just after eq. (16) in Budavari et al 2009.

    Parameters
    ----------
    t : float or array_like
        Dependent variable.
    c_sq : float, optional
        The parameter that sets how when outliers are down weighted
        (the strictness). The default is 1.
    robust_function : function, optional
        Function used to downweight outliers. The default is cauchy_like_function.

    Returns
    -------
    W_star : float or array_like
        W*=rho(t)/t

    """
    W_star = robust_function(t, c_sq) / t
    return W_star


def get_observation_vector(new_spectra, previous_mean_spectra):
    """
    Calculates the observation vector: (y = x - mean)
    (Defined just after eq. (4) in Budavari et al 2009)


    Parameters
    ----------
    new_spectra : 1d array_like
        New random spectra, given as a vector
    previous_mean_spectra : array_like
        Previous estimate of the mean spectra, given as a vector

    Returns
    -------
    observation_vector : 1d array_like
        The observation vector

    """
    observation_vector = new_spectra - previous_mean_spectra

    return observation_vector


def reconstruct_observation(observation_vector, eigen_vector_matrix):
    """
    Reconstructs observation vectors using the given eigen
    vectors and the data.

    Parameters
    ----------
    observation_vector : array_like
        A vector or matrix containing the centered data
    eigen_vector_matrix : 2d array_like matrix
        A 2d array containing eigenvectors. Each column
        (eigen_vector_matrix[:,i]) is an eigenvector
        Dimensions: (length of each specta, number of eigen vectors)

    Returns
    -------
    reconstructed_observation :
        The residuals from the observation and the model

    """
    eigen_by_transpose = np.matmul(eigen_vector_matrix, eigen_vector_matrix.T)
    reconstructed_observation = np.matmul(observation_vector, eigen_by_transpose)

    return reconstructed_observation


def get_residual(observation_vector, eigen_vector_matrix):
    """
    This function works out the residual between the observation
    vector and its reconstruction (see eq. 10 in Budavari et al 2009).

    Parameters
    ----------
    observation_vector : array_like
       Data containing the current mean subtracted spectra
    eigen_vector_matrix : 2d array_like matrix
        A 2d array containing eigenvectors. Each column
        (eigen_vector_matrix[:,i]) is an eigenvector
        Dimensions: (length of each specta, number of eigen vectors)

    Returns
    -------
    residuals : 1d array_like
        The residuals from the observation and the model

    """
    reconstruct_observation_vector = reconstruct_observation(
        observation_vector, eigen_vector_matrix
    )
    residuals = observation_vector - reconstruct_observation_vector

    return residuals


def get_mag_residual_sq(residuals):
    """
    This function works out the magnitude squared residual of a vector
    of residuals.

    Parameters
    ----------
    residuals : array_like
        The residuals from the observation and the model

    Returns
    -------
    mag_residuals_sq : float or array_like
        the squared magnitude of the given residuals

    """
    mag_residuals_sq = np.nansum(residuals**2)

    return mag_residuals_sq


def get_vqu_coeficents(robust_derivative, mag_residuals_sq):
    """
    This function calculates the coeficents for the running weights
    q, u, and v, given in Budavari et al 2009 as eq. (20), (21) and (22)
    of Budavari et al. 2009 respectivly.

    Parameters
    ----------
    robust_derivative : float
        The robust derivated evaluated at t
    mag_residuals_sq : float
        The residual representing the current spectrum


    Returns
    -------
    robust_derivative : float
        Weight for the covarience matrix

    robust_derivative * mag_residuals_sq : float
        Weight for the scale
    1 : int
        Weight for the mean

    """

    return robust_derivative, robust_derivative * mag_residuals_sq, 1


def update_vqu(vqu_prev, weights, alpha):
    """
    Calcuates the new values for q, u and v: eq. (20), (21) and (22)
    of Budavari et al. 2009 respectivly.

    Parameters
    ----------
    quv_prev : array_like
        1D array containing the three previous running totals
    weights : array_like
        1D array containing the weights for the running totals
    alpha : float
        'The forget' parameter. Value between 0 to 1. Contols how long
        previous solutions influence the current solution

    Returns
    -------
    quv : array_like
        1D array containing the new three running totals

    """
    vqu = (alpha * vqu_prev) + weights

    return vqu


def get_gammas(vqu, vqu_prev, alpha):
    """
    Calculates the gamma terms given in eq. (21), (22) and (20) of
    of Budavari et al. 2009 respectivly.
    The Gamma terms basically track the iteration number, weighted
    by the properties of the statitic.

    Parameters
    ----------
    quv : array_like
        1D array containing the new three running totals
    quv_prev : array_like
        1D array containing the three previous running totals
    alpha : float
        'The forget' parameter. Value between 0 to 1. Contols how long
        previous solutions influence the current solution

    Returns
    -------
    gamma231 : array_like
        1D array containing the gamma iterative terms

    """
    gamma123 = alpha * vqu_prev / vqu

    return gamma123


def update_scale_sq(
    scale_sq_prev,
    mag_residuals_sq,
    gamma3,
    delta=0.5,
    c_sq=1,
    robust_function=cauchy_like_function,
):
    """
    Calcuates the new estimate of the scale: eq. (19) in Budavari et al 2009.

    Parameters
    ----------
    scale_sq_prev : float
        The previous scale estimate
    mag_residuals_sq : float
        The residual representing the current spectrum
    gamma3 : float
        The interative for the scale equation
    delta : delta : float, optional
        Delta is the breakdown point (between 0 to 0.5). The default is 0.5.
    c_sq : float, optional
        The parameter that sets the strictness for the cauchy-like
        robust function. The default is 1.
    robust_function : function, optional
        Allows you to change the type of robust function used on the residuals.
        The default is the 'cauchy_like_function'.

    Returns
    -------
    scale_sq : float
        The new scale estimate

    """
    t = mag_residuals_sq / scale_sq_prev

    # Calcuating rho(t)/t here
    weight_scale = weight_star(t=t, c_sq=c_sq, robust_function=robust_function)

    scale_sq = (
        gamma3 * scale_sq_prev + (1 - gamma3) * weight_scale * mag_residuals_sq / delta
    )

    return scale_sq


def update_weighted_mean(mean_prev, observation_vector, gamma1):
    """
    Calcuates the new estimate of the location (i.e the mean): eq. (17)
    in Budavari et al 2009.

    Parameters
    ----------
    mean_prev : array_like
        Vector of the previous location estimate.
    observation_vector : array_like
        Vector of the observation_vector
    gamma1 : float
        The interative for the scale equation

    Returns
    -------
    mean_new : array_like
        Vector of the new location estimate.

    """
    mean_new = mean_prev + (1 - gamma1) * observation_vector

    return mean_new


def get_new_a(scale_sq_new, observation_vector, mag_residuals_sq, gamma2):
    """
    a is the vector appended onto the end of the eigensystem matrix A.
    a = eq. (9) in in Budavari et al 2009.

    Parameters
    ----------
    scale_sq_new : float
        The new scale estimate
    observation_vector : observation_vector : 1d array_like
        The observation vector
    mag_residuals_sq : mag_residuals_sq : float
        The residual representing the current spectrum
    gamma2 : float
        The interative for the eigensystem equation

    Returns
    -------
    a_next : array_like
        Vector of the new eigenvalue and its vectors for the new spectrum

    """
    to_sqrt = (1 - gamma2) * scale_sq_new / mag_residuals_sq
    a_next = np.sqrt(to_sqrt) * observation_vector

    return a_next


def get_A_current(eigen_vector_matrix, eigen_values, gamma2):
    """
    Calculates the current summary of the eigensystem. eq. (8)
    in Budavari et al 2009.

    Parameters
    ----------
    eigen_vector_matrix : 2d array_like matrix
        A 2d array containing eigenvectors. Each column
        (eigen_vector_matrix[:,i]) is an eigenvector
        Dimensions: (length of each specta, number of eigen vectors)
    eigen_values : array_like
        1d array since eigenvalues exist along a diagonal matrix,
        so we can just represent this as a 1d array
    gamma2 : float
        The interative for the eigensystem equation

    Returns
    -------
    A : 2d array_like matrix
        Matrix containing the summary of the eigensystem
        Dimensions: (length of each specta, number of eigen vectors)

    """
    A = np.sqrt(gamma2 * eigen_values) * eigen_vector_matrix

    return A


def get_A_new(A, a_next):
    """
    Calculates the new summary of the eigensystem. Is eq. (9) appended
    onto the end of the matrix calcuated in eq. (8) in Budavari et al 2009.

    Parameters
    ----------
    A : 2d array_like matrix
        Matrix containing the summary of the eigensystem
        Dimensions: (length of each specta, number of eigen vectors)
    a_next : array_like
        Vector of the new eigenvalue and its vectors for the new spectrum

    Returns
    -------
    A_new : 2d array_like matrix
        Matrix containing the new summary of the eigensystem
        Dimensions: (length of each specta, number of eigen vectors + 1)

    """
    A_new = np.append(A, a_next[:, None], axis=1)

    return A_new


def do_SVD_of_AtA(A_new):
    """
    Calcuates the SVD of a matrix transpose times origional matrix B.
    A * eigenvectors of B are the eigen vectors of the convarience matrix, C.
    But because B has a much smaller size than the convarience matrix,
    the speed of the SVD is massivly improved.

    Parameters
    ----------
    A_new : 2d array_like matrix
        Matrix containing the new summary of the eigensystem
        Dimensions: (length of each specta, number of eigen vectors + 1)

    Returns
    -------
    eigen_vectors_B : 2d array_like matrix
        Matrix containing the new eigen vectors
        Dimensions: (length of each specta, number of eigen vectors)
    singular_values_B : array_like
        Vector containing the singular values.

    """
    B = np.matmul(A_new.T, A_new)

    # performs SVD using LAPACK method
    # B is symetric square matrix, therefore singular values = eigenvalues
    eigen_vectors_B, singular_values_B = np.linalg.svd(B)[:-1]

    return eigen_vectors_B, singular_values_B


def update_eigen_system(eigen_vectors_B, singular_values_B, A_new, No_of_vectors):
    """
    Calculates the new eigensystem of the data covarience matrix using the
    matrix A, which is a summary of the eigensystem:
    Proof:
        Convarience matrix:
            C = A.A^T
        Columns of A are the eigen basis. Get B:
            B = A^T.A
        B is a sqaure matrix whose size is equal to the number of
        eigen vectors. We do SVD to get eigen vectors, v,
        and eigan values, s, of B:
            A^T.A v = s.v
        Left multiple by A^T:
            A.A^T.A,v = s.A.v
        But C = A.A^T, therefore A.v are the eigenvectors of C


    Parameters
    ----------
    eigen_vectors : 2d array_like matrix
        Matrix containing the eigen vectors
        Dimensions: (length of each specta, number of eigen vectors)
    singular_values : array_like
        Vector containing the singular values

    A_new : 2d array_like matrix
        Matrix containing the new summary of the eigensystem
        Dimensions: (length of each specta, number of eigen vectors + 1)

    No_of_vectors : int
        Number of eigen vectors to keep

    Returns
    -------
    new_eigen_vectors : 2d array_like matrix
        Matrix containing the final eigen vectors
    singular_values : array_like
        Vector containing the final singular values

    """
    new_eigen_vectors = np.matmul(A_new, eigen_vectors_B)

    new_eigen_vectors = new_eigen_vectors[:, :No_of_vectors]
    eigen_values = singular_values_B[:No_of_vectors]

    return new_eigen_vectors, eigen_values


def normalise_eigen_vectors(eigen_vectors, eigen_values):
    """
    Function that normalises the eigenvectors.

    Parameters
    ----------
    eigen_vectors : 2d array_like matrix
        Matrix containing the final eigen vectors
        Dimensions: (length of each specta, number of eigen vectors)
    singular_values : array_like
        Vector containing the final singular values

    Returns
    -------
    eigen_vectors_norm : 2d array_like matrix
        Matrix containing the final normalised eigen vectors
        Dimensions: (length of each specta, number of eigen vectors)

    """
    eigen_vectors_norm = eigen_vectors / np.sqrt(eigen_values)

    return eigen_vectors_norm


def bool_mask_of_bad_data(error_array):
    """
    Function that gets a mask where there are bad (=0) data points

    Parameters
    ----------
    error_array : array_like
        Array containing the error values

    Returns
    -------
    masked_data : boolean array_like
        Array of booleans

    """
    masked_data = error_array == 0

    return masked_data


def reconstruct_observation_with_scores(principle_component_scores, eigen_vectors):
    """
    Reconstruct the data using the principle component scores and
    eigenvectors.

    Parameters
    ----------
    principle_component_scores : array_like
        Array containing principle component scores.
        Dimensions: (number of spectra, number of eigen vectors)
    eigen_vectors : array_like
        Array containing the eigenvectors.
        Dimensions: (length of spectra, number of eigen vectors)

    Returns
    -------
    reconstructed_data : array_like
        Array conaining the reconstructed data

    """
    # X_expectation = P.E^T = X.E.E^T
    reconstructed_data = np.matmul(principle_component_scores, eigen_vectors.T)

    return reconstructed_data


def fill_gaps_in_data_vector(reconstructed_data, mean_array, where_bad):
    """
    Replaces the values in a spectrum with the PCA reconstructed data
    where there was bad data points.

    Parameters
    ----------
    reconstructed_data : array_like
        Array conaining the reconstructed data
        Dimensions: (length of spectra)
    mean_array : array_like
        Location estimate
        Dimensions: (length of spectra)
    where_bad : array_like
        mask where the data is bad
        Dimensions: (length of spectra)

    Returns
    -------
    filled_data_vector : array_like
        Data with bad values replaced with PCA reconstructed values

    """
    filled_data_vector = np.zeros_like(mean_array)
    filled_data_vector[where_bad] = (
        reconstructed_data[where_bad] + mean_array[where_bad]
    )

    return filled_data_vector


def normalise_filled_data_vector(spectra, filled_spectra, normalisation, where_bad):
    """
    Renormalises the data that was not replaced with PCA reconstructed values.

    Parameters
    ----------
    spectra : array_like
        Origional data spectrum
    filled_spectra : array_like
        Data spectrum where bad values replaced with PCA reconstructed values
    normalisation : array_like
        Array containting the normalisation values
    where_bad : array_like
        mask where the data is bad
    Returns
    -------
    filled_spectra : array_like
        Data with bad values replaced with PCA reconstructed values and
        the good values have been renormalised to the correct values

    """
    filled_spectra[~where_bad] = spectra[~where_bad] / normalisation

    return filled_spectra


def get_filled_observation_vector(new_spectra, eigen_vectors, mean_prev, error_array):
    """
    Wrapper function of the relevent functions needed to get the
    obeservation vector where bad values are replaced with PCA
    reconstructed values and the good values have been renormalised
    to the correct values.

    Parameters
    ----------
    new_spectra : array_like
        Origional data spectrum
    eigen_vectors : 2d array_like matrix
        Matrix containing the final eigen vectors
    mean_prev : array_like
        Location estimate
    error_array : array_like
        Error estimates of the data, where zeros represent bad pixels in the
        spectra
    Returns
    -------
    observation_vector: array_like
        array containing the data - mean
    reconstructed_data : array_like
        Array conaining the reconstructed data
    """
    where_bad = bool_mask_of_bad_data(error_array)
    # Copy to ensure we have a new memory reference
    binary_error_array = np.copy(error_array)
    binary_error_array[~where_bad] = 1

    # gappy.run_normgappy is vectorised. array[None], just adds a dummy dimension
    # so that we can run the code (array.shape = [N] array[None].shape = [1,N])
    principle_component_scores, normalisation = gappy.run_normgappy(
        error_array=binary_error_array[None],
        data=new_spectra[None],
        mean_array=mean_prev,
        eigen_vectors=eigen_vectors,
    )

    # [0] removes the dummy dimension, normalisation has two dummy dimensions
    principle_component_scores = principle_component_scores[0]
    normalisation = normalisation[0][0]

    reconstructed_spectra = reconstruct_observation_with_scores(
        principle_component_scores, eigen_vectors
    )
    filled_spectra = fill_gaps_in_data_vector(
        reconstructed_spectra, mean_prev, where_bad
    )
    filled_spectra = normalise_filled_data_vector(
        new_spectra, filled_spectra, normalisation, where_bad
    )

    observation_vector = get_observation_vector(
        new_spectra=filled_spectra, previous_mean_spectra=mean_prev
    )

    return observation_vector, reconstructed_spectra


def iterate_PCA_with_data_gaps(
    eigen_system_dict,
    new_spectra,
    alpha,
    error_array,
    delta=0.5,
    robust_function=cauchy_like_function,
    robust_derivative=derivate_of_cauchy_like_function,
    c_sq=1,
):
    """
    Main wrapper function for calculating robust pca with data that contains
    bad data points, or "gaps". The Gaps should be entered as zeros in the
    error array.

    Parameters
    ----------
    eigen_system_dict : dictionary
        Dictionary containing the eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu).
    new_spectra : array_like
        Current spectrum to iterate the PCA with
    alpha : float
        'The forget' parameter. Value between 0 to 1. Contols how long
        previous solutions influence the current solution
    error_array : array_like
        Error estimates of the data, where zeros represent bad pixels in the
        spectra
    delta : float, optional
        Delta is the breakdown point (between 0 to 0.5). The default is 0.5.
    robust_function : function, optional
        Function used to downweight outliers. The default is cauchy_like_function.
    robust_derivative : function, optional
        Derivative of the function used to downweight outliers.
        The default is derivate_of_cauchy_like_function.
    c_sq : float, optional
        Parameter for setting when the robust function downweights
        outliers. The default is 1.

    Returns
    -------
    eigen_system_dict : dictionary
        Dictionary containing the updated eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu).

    """
    eigen_vectors = eigen_system_dict["U"]  # eigenvectors as (nbin,nvec) array
    mean_prev = eigen_system_dict["m"]  # mean (nbin) vector

    observation_vector, reconstructed_spectra = get_filled_observation_vector(
        new_spectra, eigen_vectors, mean_prev, error_array
    )
    residuals = observation_vector - reconstructed_spectra

    eigen_system_dict = PCA_from_residuals(
        eigen_system_dict,
        residuals,
        observation_vector,
        alpha,
        delta=delta,
        robust_function=robust_function,
        robust_derivative=robust_derivative,
        c_sq=c_sq,
    )

    return eigen_system_dict


def iterate_PCA(
    eigen_system_dict,
    new_spectra,
    alpha,
    error_array=None,
    delta=0.5,
    robust_function=cauchy_like_function,
    robust_derivative=derivate_of_cauchy_like_function,
    c_sq=1,
):
    """
    Main wrapper function for calculating robust pca with data without
    bad data points, or "gaps".

    Parameters
    ----------
    eigen_system_dict : dictionary
        Dictionary containing the eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu).
    new_spectra : array_like
        Current spectrum to iterate the PCA with
    alpha : float
        'The forget' parameter. Value between 0 to 1. Contols how long
        previous solutions influence the current solution
    error_array : None, optional
        Just a dummy variable that doesn't do anything when calling
        this wrapper
    delta : float, optional
        Delta is the breakdown point (between 0 to 0.5). The default is 0.5.
    robust_function : function, optional
        Function used to downweight outliers. The default is cauchy_like_function.
    robust_derivative : function, optional
        Derivative of the function used to downweight outliers.
        The default is derivate_of_cauchy_like_function.
    c_sq : float, optional
        Parameter for setting when the robust function downweights
        outliers. The default is 1.

    Returns
    -------
    eigen_system_dict : dictionary
        Dictionary containing the updated eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu).
    """
    # eigen_vectors. Columns are the eigen vectors
    eigen_vectors = eigen_system_dict["U"]
    mean_prev = eigen_system_dict["m"]

    observation_vector = get_observation_vector(
        new_spectra=new_spectra, previous_mean_spectra=mean_prev
    )

    residuals = get_residual(
        observation_vector=observation_vector, eigen_vector_matrix=eigen_vectors
    )

    eigen_system_dict = PCA_from_residuals(
        eigen_system_dict,
        residuals,
        observation_vector,
        alpha,
        delta=delta,
        robust_function=robust_function,
        robust_derivative=robust_derivative,
        c_sq=c_sq,
    )

    return eigen_system_dict


def PCA_from_residuals(
    eigen_system_dict,
    residuals,
    observation_vector,
    alpha,
    delta=0.5,
    robust_function=cauchy_like_function,
    robust_derivative=derivate_of_cauchy_like_function,
    c_sq=1,
):
    """
    Wrapper function to calculate one iteration of robust PCA.

    Parameters
    ----------
    eigen_system_dict : dictionary
        Dictionary containing the eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu).
    residuals : array_like
        The residuals of the data to the pca reconstructed data
    observation_vector: array_like
        array containing the data - mean
    alpha : float
        'The forget' parameter. Value between 0 to 1. Contols how long
        previous solutions influence the current solution
    error_array : None, optional
        Just a dummy variable that doesn't do anything when calling
        this wrapper
    delta : float, optional
        Delta is the breakdown point (between 0 to 0.5). The default is 0.5.
    robust_function : function, optional
        Function used to downweight outliers. The default is cauchy_like_function.
    robust_derivative : function, optional
        Derivative of the function used to downweight outliers.
        The default is derivate_of_cauchy_like_function.
    c_sq : float, optional
        Parameter for setting when the robust function downweights
        outliers. The default is 1.

    Returns
    -------
    eigen_system_dict : dictionary
        Dictionary containing the updated eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu).
    """
    # for now, kept the same dictionary names as VW ILD implimentation
    eigen_vectors = eigen_system_dict["U"]  # eigenvectors as (nbin,nvec) array
    eigen_values = eigen_system_dict["W"]  # eigenvalues (nvec) vector
    mean_prev = eigen_system_dict["m"]  # mean (nbin) vector
    vqu_prev = eigen_system_dict["vqu"]  # running weights 1x3 array
    sigma_sq = eigen_system_dict["sig2"]  # scale float

    mag_residual_sq = get_mag_residual_sq(residuals=residuals)
    weight1 = robust_derivative(t=mag_residual_sq / sigma_sq, c_sq=c_sq)

    weight_coefficants = get_vqu_coeficents(
        robust_derivative=weight1, mag_residuals_sq=mag_residual_sq
    )

    vqu_new = update_vqu(vqu_prev=vqu_prev, weights=weight_coefficants, alpha=alpha)

    gammas123 = get_gammas(vqu=vqu_new, vqu_prev=vqu_prev, alpha=alpha)

    eigen_system_dict["vqu"] = vqu_new

    # Must update mean and scale BEFORE updating A matrix
    new_mean = update_weighted_mean(
        mean_prev=mean_prev, observation_vector=observation_vector, gamma1=gammas123[0]
    )
    eigen_system_dict["m"] = new_mean

    new_scale_sq = update_scale_sq(
        scale_sq_prev=sigma_sq,
        mag_residuals_sq=mag_residual_sq,
        gamma3=gammas123[2],
        delta=delta,
        c_sq=c_sq,
        robust_function=robust_function,
    )

    eigen_system_dict["sig2"] = new_scale_sq

    new_a = get_new_a(
        scale_sq_new=new_scale_sq,
        observation_vector=observation_vector,
        mag_residuals_sq=mag_residual_sq,
        gamma2=gammas123[1],
    )

    A = get_A_current(
        eigen_vector_matrix=eigen_vectors,
        eigen_values=eigen_values,
        gamma2=gammas123[1],
    )

    A_new = get_A_new(A=A, a_next=new_a)

    # singular_values_B are the eigenvalues since B is a square array
    eigen_vectors_B, singular_values_B = do_SVD_of_AtA(A_new=A_new)

    eigen_vectors_new, eigen_values_new = update_eigen_system(
        eigen_vectors_B=eigen_vectors_B,
        singular_values_B=singular_values_B,
        A_new=A_new,
        No_of_vectors=len(eigen_values),
    )

    eigen_vectors_new_normed = normalise_eigen_vectors(
        eigen_vectors=eigen_vectors_new, eigen_values=eigen_values_new
    )

    eigen_system_dict["U"] = eigen_vectors_new_normed
    eigen_system_dict["W"] = eigen_values_new

    return eigen_system_dict


def main():
    pass


if __name__ == "__main__":
    main()
