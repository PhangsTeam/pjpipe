# -*- coding: utf-8 -*-
"""
Copyright (C) 2007 Tamas Budavari and Vivienne Wild (MAGPop)

This script contains the functions needed to initalise the parameters
and eigen system to run the robust itrative PCA technique from Budavari
et al. 2009 (MNRAS 394, 1496–1502). Each method has been written out
separately but the entire method can be run using the wrapper function
provided. To test, you can just run the wrapper function entering only your
dataset (dimensions must be [number of spectra, length of spectra]) with
the preset options.
WARNING: Assumes the data entered is not sorted or grouped in any way (i.e.,
random oder of spectra)

NOTES:
    The inital mean is based of the entire dataset, rather than the initalise
    amount
    The inital eigen vectors and values are calcuated using the entire dataset
    rather than the initalise amount
"""


import numpy as np
import LEGACY.pca.vwpca_inc as vwpca

def mean_subtracted_data(data):
    """
    Calculates the mean and subtracts it from each spectra.

    Parameters
    ----------
    data : array_like
        The data to centre
        Dimentions: [number of spectra, length of spectra]

    Returns
    -------
    centred_data : array_like
        Data centered using its mean.
        Dimentions: [number of spectra, length of spectra]
    vector_mean : array like
        The mean data vector
        Dimentions: [length of spectra]

    """
    vector_mean = np.nanmean(data, axis=0)
    centred_data = data - vector_mean

    return centred_data, vector_mean

def do_initial_SVD(data):
    """
    Runs SVD on a data array to calculate the eigen vectors transpose
    and singular values. (singular values are the positive root of the
    eigen values).
    WARNING: The eigen vectors returned are transposed.

    Parameters
    ----------
    data : array_like
        Centered normalised data
        Dimentions: [number of spectra, length of spectra]

    Returns
    -------
    eigen_vectors_T : array_like
        Eigen vectors transposed representing the data
        Dimensions: [length of spectra, length of spectra]
    singular_values : 1d array
        Singular values of the SVD decomposition

    """
    singular_values, eigen_vectors_T =  np.linalg.svd(data)[1:]
    return eigen_vectors_T, singular_values

def get_eigen_system(data):
    """
    Method for calculating the eigen vectors and values of the data using
    SVD.

    Parameters
    ----------
    data : array_like
        Centered normalised data
        Dimentions: [number of spectra, length of spectra]

    Returns
    -------
    eigen_vectors_T.T: array_like
        Eigen vectors representing the data
        Dimensions: [length of spectra, length of spectra]
    eigen_values : 1d array
        Eigen values of data
    """
    eigen_vectors_T, singular_values = do_initial_SVD(data=data)
    eigen_values = singular_values**2

    # array.T performs a matrix transpose
    return eigen_vectors_T.T, eigen_values

def initalise_eigensystem(data_centred):
    """
    Method wrapping around the data to calcuate the eigen vectors and values
    Data needs to be normalise, which is done here.

    Parameters
    ----------
    data_centred : array_like
        Data that has been centred

    Returns
    -------
    eigen_vectors: array_like
        Eigen vectors representing the data
        Dimensions: [length of spectra, length of spectra]
    eigen_values : 1d array
        Eigen values of data

    """
    amount_of_data_vectors = data_centred.shape[0]
    data_normed = data_centred /  np.sqrt(amount_of_data_vectors)

    eigen_vectors, eigen_values = get_eigen_system(data=data_normed)

    return eigen_vectors, eigen_values

def reconstruct_observation(eigen_by_transpose, observation_vector):
    """
    Reconstructs observation vectors using the given eigen
    vectors and the data.

    Parameters
    ----------
    observation_vector : array_like
        A vector or matrix transpose containing the current
        mean subtracted spectra
    eigen_vector_matrix : 2d array_like matrix
        A 2d array containing eigenvectors. Each column
        (eigen_vector_matrix[:,i]) is an eigenvector
        Dimensions: [length of spectra, number of eigen vectors]

    Returns
    -------
    reconstructed_observation :
        The residuals from the observation and the model

    """
    reconstructed_observation = np.matmul(observation_vector, eigen_by_transpose)

    return reconstructed_observation



def get_residual(observation_vector, eigen_vector_matrix):
    """
    This function works out the residual between the observation
    vector and its reconstruction (see eq. 10 in Budavari et al 2009).
    Here is is the matrix version of the calculation.

    Parameters
    ----------
    observation_vector : array_like
        A vector or matrix containing the current
        mean subtracted data
    eigen_vector_matrix : 2d array_like matrix
        A 2d array containing eigenvectors. Each column
        (eigen_vector_matrix[:,i]) is an eigenvector
        Dimensions: [length of spectra, number of eigen vectors]

    Returns
    -------
    residuals : array_like
        The residuals from the observation and the model

    """
    #multiplying the eigenvectors by its transpositons: E.E^T
    eigen_by_transpose = np.matmul(eigen_vector_matrix, eigen_vector_matrix.T)

    #reconstructing data using eigensysten: y(E.E^T),
    reconstruct_observation_vector = reconstruct_observation(eigen_by_transpose, observation_vector)

    residuals = observation_vector - reconstruct_observation_vector

    return residuals

def mag_residual_sq(residuals, error_map=None):
    """
    This function works out the magnitude squared residual of a vector
    of residuals. Ifan error map is given, were zeros represent
    bad data points, the sum excludes the bad data points.

    Parameters
    ----------
    residuals : array_like
        The residuals from the observation and the model
    error_map: None type or array_like, optional

    Returns
    -------
    mag_residuals_sq : float or array_like
        the squared magnitude of the given residuals

    """
    if error_map is not None:
        error_map = error_map[:residuals.shape[0]]
        residuals[error_map==0] = np.nan

    mag_residuals_sq = np.nansum(residuals**2, axis=1)

    return mag_residuals_sq

def get_mag_residuals_sq(data, eigen_vectors, error_map=None):
    """
    This function works out the magnitude squared residuals of a vector, or
    matrix of residuals.

    Parameters
    ----------
    data : array_like
        Centered data
        Dimentions: [number of spectra, length of spectra]
    eigen_vectors : 2d array_like matrix
        A 2d array containing eigenvectors. Each column
        (eigen_vector_matrix[:,i]) is an eigenvector
        Dimensions: [length of spectra, number of eigen vectors]


    Returns
    -------
    mag_residuals_sq : float or array_like
        The squared magnitude of the given residuals

    """
    residuals_transpose = get_residual(data, eigen_vectors)
    mag_residuals_squared = mag_residual_sq(residuals_transpose, error_map=error_map)

    return mag_residuals_squared


def initalise_scale_using_residuals_sq(residuals_sq, breakdown_point=0.5, amount_of_eigen=10,
                                       robust_function=vwpca.cauchy_like_function, c_sq=1):
    """
    A function that provides an inital guess of the scale squard (sigma squared)
    following the description in section 3.2 of Budavari et al. 2009


    Parameters
    ----------
    residuals_sq : array_like
        Residual of each spectrum minus the reconstructed spectrum
    breakdown_point : float, optional
        Breakdown point (between 0 to 0.5) sets the robustness of the statistics
        The lower the number, the faster, but less robust the result will be.
        The default is 0.5.
    amount_of_eigen : int, optional
        Number of eigen vectors to use. The default is 10.
    robust_function : function, optional
        Function used to downweight outliers. The default is vwpca.cauchy_like_function.
    c_sq : float, optional
        Parameter for setting when the robust function downweights
        outliers. The default is 1.

    Returns
    -------
    scale_sq : float
        Scale squared estimator

    """

    scale_sq = np.nanmean(residuals_sq)

    for i in range(amount_of_eigen):

        t = residuals_sq/scale_sq
        robust_weighting = robust_function(t=t, c_sq=c_sq)
        scale_sq *= np.nanmean(robust_weighting) / breakdown_point

    return scale_sq


def initalise_vqu(residuals_sq, scale_sq, robust_derivative=vwpca.derivate_of_cauchy_like_function, c_sq=1):
    """
    Initalise coeficents for the running weights q, u, and v,
    given in Budavari et al 2009 (see equations (20), (21) and (22).

    Parameters
    ----------
    residuals_sq : array_like
        Residual of each spectrum minus the reconstructed spectrum
    scale_sq : float
        Scale squared estimator
    robust_derivative : function, optional
        Derivative of the function used to downweight outliers. The default
        is vwpca.derivate_of_cauchy_like_function.
    c_sq : float, optional
        Parameter for setting when the robust function downweights
        outliers. The default is 1.

    Returns
    -------
    inital_quv : array_like
        1D array containing the inital three running totals used to update the
        iterative statistics

    """
    t = residuals_sq/scale_sq
    weight1 = robust_derivative(t=t, c_sq=c_sq)

    weight_coefficants_3 = np.ones_like(weight1)

    vqu_vectors = np.column_stack([weight1,
                                   weight1*residuals_sq,
                                   weight_coefficants_3])

    inital_quv = np.nanmean(vqu_vectors, axis=0)

    return inital_quv

def input_for_pca(mean_array, eigen_values, eigen_vectors, scale_sq, vqu):
    """
    Initalise the eigen system that will get updated.

    Parameters
    ----------
    mean_array : array_like
        The inital mean data vector
    eigen_values : array_like
        The inital eigan values of the data
    eigen_vectors : array_like
        The inital eigen vectors of the data
        Dimensions: [length of spectra, number of eigen vectors]
    scale_sq : float
        Inital estimate of the scale squared
    vqu : arra_like
        1D array containing the intal three running totals used to update the
        iterative statistics.

    Returns
    -------
    eigen_system_dict : dictionary
        Dictionary containing the inital eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu).

    """
    eigen_system_dict = {'U':eigen_vectors, 'm':mean_array,
                         'W':eigen_values,  'vqu':vqu,
                         'sig2':scale_sq}

    return eigen_system_dict

def forget_parameter(amount_of_spectra, memory=1):
    """
    Method used to set the forget parameter, alpha, following
    Budavari et al. 2009 in section 3.2.

    Parameters
    ----------
    amount_of_spectra : int
        number of spectra in data set.
    memory : float, optional
        Allows one to vary the forget parameter while still basing alpha of
        the number of spectra. The default is 1.

    Returns
    -------
    forget_param : float
        Parameter which set when previous solutions are downweighted

    """
    forget_param = 1 - 1 / (amount_of_spectra * memory)

    return forget_param


def initalise_robust_pca_param(data, errors=None, amount_of_eigen=10, amount_to_initalise=200, breakdown_point=0.5,
                               robust_function=vwpca.cauchy_like_function, robust_derivative=vwpca.derivate_of_cauchy_like_function,
                               c_sq=1):
    """
    Wrapper function to initalise the eigensystem all in one place using
    the functions in this script. If an error map is given, the initalised
    eigen system takes bad data points (set to zero in the error array) into
    account when initalising the eigen system for the robust PCA

    Parameters
    ----------
    data : array_like
        Unordered data matrix
        Dimensions:  [Number of spectra, length of spectra]
    errors : None type or array_like, optional
        Array containing the data errors, where zero indicates where the
        data is bad and needs replacing. If None, assumes all the data is valid
        The default is None.
    amount_of_eigen : int, optional
        Number of eigen vectors to keep. The default is 10.
    amount_to_initalise : in, optional
        amount of data vectors to be used when initalising the eigen system
        for the robust PCA. The default is 200.
    breakdown point : Between 0 to 0.5. Sets the robustness of the statistics
        The lower the number, the faster, but less robust the result will be.
        The default is 0.5.
    robust_function : function, optional
        Function used to downweight outliers. The default is
        vwpca.cauchy_like_function.
    robust_derivative : function, optional
        Derivative of the function used to downweight outliers.
        The default is vwpca.derivate_of_cauchy_like_function.
    c_sq : float, optional
        Parameter for setting when the robust function downweights
        outliers. The default is 1.

    Returns
    -------
    eigen_system_dict : dict
        Dictionary containing the inital eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu) needed to
        start the robust PCA method

    """
    data_centred, mean_initial = mean_subtracted_data(data=data)

    eigen_vectors_initial, eigen_values_inital = initalise_eigensystem(data_centred=data_centred)

    eigen_vectors_initial = eigen_vectors_initial[:,:amount_of_eigen]
    eigen_values_initial = eigen_values_inital[:amount_of_eigen]

    residuals_sq = get_mag_residuals_sq(data=data_centred[:amount_to_initalise],
                                    eigen_vectors=eigen_vectors_initial, error_map=errors)

    scale_sq_initial = initalise_scale_using_residuals_sq(residuals_sq=residuals_sq,
                                                       breakdown_point=breakdown_point,
                                                       amount_of_eigen=10,
                                                       robust_function=robust_function,
                                                       c_sq=c_sq)

    vqu_initial = initalise_vqu(residuals_sq=residuals_sq,
                                scale_sq=scale_sq_initial,
                                robust_derivative=robust_derivative,
                                c_sq=c_sq)


    eigen_system_dict = input_for_pca(mean_array=mean_initial,
                                      eigen_values=eigen_values_initial,
                                      eigen_vectors=eigen_vectors_initial,
                                      scale_sq=scale_sq_initial, vqu=vqu_initial)
    return eigen_system_dict

def run_robust_pca(data, errors=None, amount_of_eigen=10, amount_to_initalise=200,
               number_of_iterations=5, forget_param=None,
               breakdown_point=0.5, memory=1, save_extra_param=False, c_sq=1):
    """
    Main wrapper to perform the robust pca method on an entire dataset.

    Parameters
    ----------
    data : array_like
        Unordered data matrix
        Dimensions:  [Number of spectra, length of spectra]
    errors : None type or array_like, optional
        Array containing the data errors, where zero indicates where the
        data is bad and needs replacing. If None, assumes all the data is valid
        The default is None.
    amount_of_eigen : int, optional
        Number of eigen vectors to keep. The default is 10.
    amount_to_initalise : in, optional
        amount of data vectors to be used when initalising the eigen system
        for the robust PCA. The default is 200.:
    number_of_iterations : int, optional
        Number of times to run the robust pca method. The default is 5.
    forget_param : float, optional
        Parameter which set when previous solutions are downweighted.
        The default is None.
    breakdown point : Float, optional
        Between 0 to 0.5. Sets the robustness of the statistics
        The lower the number, the faster, but less robust the result will be.
        The default is 0.5.
    memory : float, optional
        Allows one to vary the forget parameter while still basing alpha of
        the number of spectra. The default is 1.
    save_extra_param : Boolean, optional
        Switch determining nature of return value. When False just
        the eigen system is return. When true extra parameters to track
        how the robust PCA changes for each incriment is returned.
        The default is False.

    Returns
    -------
    eigen_system_dict : dict
        Dictionary containing the final eigenbasis, location (i.e., mean),
        scale squared (sigma squared) and robust weights (vqu) of the data
    tracked_eigen_system_dict : dictionary
    Present only if `save_extra_param` = True. Contains extra parameters to track
        how the robust PCA changes for each incriment. Tracks the eigen
        values, the v weights and scale squared

    """
    amount_of_spectra, number_of_pixels = data.shape

    if forget_param is None:
        forget_param = forget_parameter(amount_of_spectra=amount_of_spectra,
                                        memory=memory)

    eigen_system_dict = initalise_robust_pca_param(data=data,
                                                   errors=errors,
                                                   amount_of_eigen=amount_of_eigen,
                                                   amount_to_initalise=amount_to_initalise,
                                                   breakdown_point=breakdown_point)

    if errors is not None:
        pca_function = vwpca.iterate_PCA_with_data_gaps
    else:
        pca_function = vwpca.iterate_PCA
        # Makes a None list of length `amount_of_spectra`
        errors = [None] * amount_of_spectra

    counter = 0
    save_amount = amount_of_spectra * number_of_iterations - amount_to_initalise + 1
    tracked_eigen_system_dict = {}

    #Need to skip the data we used to initalise the eigenbasis
    start = amount_to_initalise

    for k in range(number_of_iterations):
        for sp in range(start, amount_of_spectra):
            tracked_eigen_system_dict, counter = track_eigensystem_updates(counter, eigen_system_dict,
                                                                           tracked_eigen_system_dict,
                                                                           save_amount)

            eigen_system_dict = pca_function(eigen_system_dict=eigen_system_dict, new_spectra=data[sp],
                                                  alpha=forget_param, error_array=errors[sp], delta=breakdown_point, c_sq=c_sq)
        start = 0
        # after one iteration of the entire dataset, the past is forgotten
        # so we can now include the data we used to initalise the inital
        # eigen basis

    if save_extra_param:
        return eigen_system_dict, tracked_eigen_system_dict
    else:
        return eigen_system_dict

def track_eigensystem_updates(counter, eigen_system_dict, tracked_eigen_system_dict, save_amount):
    """
    Updates the extra variables used to track the evolution of the eigen
    system as the PCA incriments.

    Parameters
    ----------
    counter : int
        Current count of the incriment
    eigen_system_dict : Dictionary containing the inital eigenbasis,
    location (i.e., mean), scale squared (sigma squared) and robust
    weights (vqu) for the robust PCA
    tracked_eigen_system_dict : dict
        Contains extra parameters tracking how the robust PCA changes for each
        incriment. Tracks the eigen values, the `v` weights and scale squared
    save_amount : int
        Total number of incriments.

    Returns
    -------
    tracked_eigen_system_dict : dict
        Contains extra parameters tracking how the robust PCA changes for each
        incriment. Tracks the eigen values, the `v` weights and scale squared
    counter + 1 : int
        Next count of the incriment.

    """
    if counter == 0:
        amount_of_eigen = len(eigen_system_dict['W'])
        tracked_eigen_system_dict = initialise_tracked_eigen_updates(save_amount, amount_of_eigen)

    tracked_eigen_system_dict['W'][counter] = eigen_system_dict['W']
    tracked_eigen_system_dict['vqu'][counter] = eigen_system_dict['vqu'][0]
    tracked_eigen_system_dict['sig2'][counter] = eigen_system_dict['sig2']

    return tracked_eigen_system_dict, counter + 1

def initialise_tracked_eigen_updates(save_amount, amount_of_eigen):
    """
    Initaliaises the dictionary to track the evolution of the eigen
    system as the PCA incriments.

    Parameters
    ----------
    save_amount : int
        Total number of incriments.
    amount_of_eigen : int
        Number of eigen vectors.

    Returns
    -------
    tracked_eigen_system_dict : dict
        Initalised dictonarytracking how the robust PCA changes for each
        incriment. Tracks the eigen values, the `v` weights and scale squared

    """
    eigenvalues_interated = np.zeros([save_amount, amount_of_eigen])
    vk_interated = np.zeros(save_amount)
    scalesq_interated = np.zeros(save_amount)

    tracked_eigen_system = {'W':eigenvalues_interated, 'sig2':scalesq_interated, 'vqu':vk_interated}

    return tracked_eigen_system


def main():
    pass

if __name__ == "__main__":
    main()



