# -*- coding: utf-8 -*-
"""
Copyright (C) 2007 Tamas Budavari and Vivienne Wild (MAGPop)

This script contains the vectorised functions for calcuating principle
component scores when some of the data is bad/missing based off
Connolly & Szalay (1999, AJ, 117, 2052). When data is missing,
the assumption of Euclidean spaced data points is broken. To get around this,
we can use the expectation values (reconstrcted data using eigen system)
to replace the bad values with a reconstruction of the data. By doing this,
we need to also renormalise the data after. The cost is that the eigen vectors
are no longer entirally othogonal.

----------------------------------
So we minimise:
    X_sq = W(|x - Va|^2)

with respect to a where:
`W` is the window function.
`a` are the pc scores (Dimensions [, number of eigen vectors])
`V` are the eigen vectors (Dimensions [length of data vector, number of eigen vectors])
`x` is the data with gaps (Dimensions [, length of data vector])

So the weights, `w`, in normgapy are used as a sudo window function,
where zeros are skipped and what we replace all other values
of w are set to 1 and are these values are renormlised at the end

The maths behind the normalisation are in Lemson [in prep from 2008].
Could not find the paper. So I have worked out the maths here:

So to normalise when we replace gaps, we introduce the normalisation constant
on the reconstructed data (called `N`, hense why vwpca_normgappy
has the coment: solve partial chi^2/partial N = 0)

So we write the reconstruction as : N(m +Va)

`m` is the mean

we minimise:
    X_sq =  W(|x - N(m + Va)|^2)

with respect to `a` and `N`.

V is out eigen vector matrix
x is the data vector (or a matrx with dimentsions [Length of data, 1])
m is the data vector (or a matrx with dimentsions [Length of data, 1])
a is a the pc scores vector (or a matrx with dimentsions [number of eigen, 1])

For brevity, I have removed the Window function from
this maths walk through but it is still there for every variable. When we
minimise, we are looking to find a solution for `a` that is independent
from `N`. So to minimise, we partially differentiate `X_sq` with respect
to `a` and `N`, and set the solution to zero:


dX_sq/da = 0 = V^T*x - NV^T*m - NV^T*V*a

V^T*x are the windowed pca scores. VW call it `F`
V^T*m are the windowed pca scores of the average data spectrum. VW call it `E`
V^T*V are the windowed eigen vectors transpose by itself. VW call it `M`

dX_sq/dN = 0 = x^T*m - N*m^T*m - N*m^T*V*a + a^T*dX_sq/da

x^T*m is a windowed dot product (x.m) of the data to the mean so is scalar.
WV call this `Fpr`
m^T*m is a windowed dot product of mean. Is scalar. WV call this `Mpr`

Want to elimante N: Can multiply dX_sq/da by x^T*m=Fpr and multiply
V^T*x=F by first three terms of dX_sq/dN

: dX_sq/da * Fpr  = F*Fpr - Fpr*NV^T*m - Fpr*NV^T*V*a = 0

: F*(x^T*m - N*m^T*m - N*m^T*V*a) = F*Fpr - N*F*m^T*m - N*F*m^T*V*a = 0

Now we can subract them from eachother, which eliminates Fpr*F. Now all terms
have and `N`, which we can eliminate, leaving use with only `a`. We
have achieved our goal and minimised `X_sq`, with an independent solution
for a (and therefore for N)

- F*m^T*m - F*m^T*V*a + Fpr*V^T*m + Fpr*V^T*V*a = 0

Factorise out the a:

(Fpr*NV^T*V - F*m^T*V)a = F*m^T*m - Fpr*V^T*m

Call Fpr*V^T*V - F*m^T*V = M_new
Call F*m^T*m - Fpr*V^T*m = F_new

M_new*a = F_new
a = M_new^-1 * F_new

Use dX_sq/da to now get N

0 = F - NV^T*m - NV^T*V*a

N(V^T*m + NV^T*V*a) = F
N = F / (V^T*m + NV^T*V*a)

"""

import numpy as np

def get_weights(error_array):
    """
    Gets the weights for the data. If we do not consider noise, weights is a
    window function, where 1 is where we have good data and 0 is where we have
    bad data.

    Parameters
    ----------
    error_array : array_like
        Error array for the entire data set
        Dimensions: [number of data vectors, length of data vector]

    Returns
    -------
    weights : array_like
        Weights for the data
        Dimensions: [number of data vectors, length of data vector]

    """
    weights = 1 / error_array

    weights[np.isinf(weights)] = 0 #np.nan < matmul doesn't work with nans

    return weights


def where_data_all_good(error_array):
    """
    Finds and excludes where an entire data vector has bad data points

    Parameters
    ----------
    error_array : array_like
        error array for the entire data set
        Dimensions: [number of data vectors, length of data vector]

    Returns
    -------
    bad_indices : array_like
        A 1D array containing the positions excluding any data vectors that
        only contain bad values.

    """
    error_sum = np.nansum(error_array, axis=1)

    good_indices = np.where(error_sum!=0)[0]

    return good_indices

def dot_product_vectorised(array_1, array_2):
    """
    Sudo dot product by multiplying 2d arrays and summing along the rows.

    Parameters
    ----------
    array_1 : array_like
        First array to sum
    array_2 : array_like
        Second array to sum

    Returns
    -------
    array_1_dot_array_2 : array_like
        Result of sudo dot product

    """
    array_1_dot_array_2 = np.nansum(array_1 * array_2 , axis=1)

    return array_1_dot_array_2


def get_dchi_sq_dnorm(weights, data, mean_array, eigen_vectors):
    """
    Wrapper to calculate the terms of dX_sq/dN = 0
    = W(x^T*m) - N*W(m^T*m) - N*W(m^T*V*a) + a^T*W(dX_sq/da)
    where `W` is a window function (`weights`), `N` is the normalisation factor
    `a` are the pc scores, `x` is the data, `m` is the mean.
    This function calculates the first three terms:

      1.  W(x^T*m)
      2.  W(m^T*m)
      3.  W(m^T*V)

    Parameters
    ----------
    weights : array_like
        Weights for the data
        Dimensions: [number of data vectors, length of data vector]
    data : array_like
        Data with bad values.
        Dimensions: [number of data vectors, length of data vector]
    mean_array : array_like
        Average data vector
        Dimensions: [length of data vector]
    eigen_vectors : array_like
        eigen vectors describing `data`
        Dimensions: [length of spectra, number of eigen vectors]

    Returns
    -------
    mean_dot_data : array_like
    Mean dotted by the data, excluding bad values.
        Dimensions: [number of data vectors]
    mean_dot_mean : array_like
        Mean dotted by itself, excluding bad values.
        Dimensions: [number of data vectors]
    mean_scores : array_like
        mean projected to eigen system
        Dimensions: [number of data vectors, number of eigen vectors]

    """
    weighted_mean = weights * mean_array

    mean_dot_data = dot_product_vectorised(weighted_mean, data)
    mean_dot_mean = dot_product_vectorised(weighted_mean, mean_array)

    eigen_vectors_3d = np.ones([data.shape[0], *eigen_vectors.shape]) * eigen_vectors
    ###
    # Each weight*mean spectrum is multiplied with the eigen vectors.
    # we then sum along the length of the data, so we end up with a matrix
    # with dimensions: [number of data vectors, number of eigen vectors]
    # When not vectorised (so done one data vector at a time, `mean_scores`
    # has length number of eigen vectors)
    ###
    mean_scores = dot_product_vectorised(weighted_mean[:,:,None], eigen_vectors_3d)


    return mean_dot_data, mean_dot_mean, mean_scores


def get_dchi_sq_dscores(weights, data, eigen_vectors):
    """
    dX_sq/da = 0 = W(V^T*x) - NW(V^T*V*a) - NW(V^T*m)
    where `W` is a window function (`weights`), `N` is the normalisation factor
    `a` are the pc scores, `x` is the data, `m` is the mean
    This function calculates the first two terms, where the third term
    is calculated and given in dX_sq/dN:

      1.  W(V^T*x)
      2.  W(V^T*V)

    Parameters
    ----------
    weights : array_like
        Weights for the data
        Dimensions: [number of data vectors, length of data vector]
    data : array_like
        Data with bad values.
        Dimensions: [number of data vectors, length of data vector]
    eigen_vectors : array_like
        eigen vectors describing `data`
        Dimensions: [length of spectra, number of eigen vectors]

    Returns
    -------
    weighted_scores : array_like
        PC scores without bad data points
        Dimensions: [number of data vectors, length of eigen vector]
    weighted_eigen_transpose_by_eigen: array_like
        weighted eigen vectors transposed by eigen vectors
        Dimensions: [number of data vectors, number of eigen vectors, number of eigen vectors]

    """
    weighted_data = weights * data
    weighted_scores = np.matmul(weighted_data, eigen_vectors)

    # None adds a dummy dimension to allow for the vectorised operation
    weighted_eigen_vectors = weights[:,:,None] * eigen_vectors
    weighted_eigen_transpose_by_eigen = np.matmul(eigen_vectors.T, weighted_eigen_vectors)

    return weighted_scores, weighted_eigen_transpose_by_eigen


def get_ammended_weigthed_correlation_matrix(mean_dot_data, weighted_eigen_transpose_by_eigen, weighted_scores, mean_scores):
    """
    Without the normalisation factor, the correlation matrix is
    'weighted_eigen_transpose_by_eigen'^-1. With the normalisation factor this
    matrix has an ammended result given here.

    Parameters
    ----------
    mean_dot_data : array_like
    Mean dotted by the data, excluding bad values.
        Dimensions: [number of data vectors]
    weighted_eigen_transpose_by_eigen: array_like
        weighted eigen vectors transposed by eigen vectors
    weighted_scores : array_like
        PC scores without bad data points
        Dimensions: [number of data vectors, number of eigen vectors]
    mean_scores : array_like
        Mean projected to eigen system (i.e., pc scores of the mean)
        Dimensions: [number of data vectors, number of eigen vectors]

    Returns
    -------
    new_correlation_matrix : array_like
        new_correlation_matrix^-1 are the correlation of the weighted eigen vectors
        Dimensions: [number of data vectors, number of eigen vectors, number of eigen vectors]

    """
    M1 = mean_dot_data[:,None,None] * weighted_eigen_transpose_by_eigen
    #vectorised
    M2 = np.tensordot(weighted_scores, mean_scores, axes=0)[0,:,0]

    new_correlation_matrix = M1 - M2

    return new_correlation_matrix

def get_ammended_weighted_scores(mean_dot_mean, weighted_scores, mean_dot_data, mean_scores):
    """
    Without the normalisation factor, the weighted pc scores are just
    'weighted_scores'. With the normalisation factor, this vector
    matrix has an ammended result given here.

    Parameters
    ----------
    mean_dot_mean : array_like
        Mean dotted by itself, excluding bad values.
        Dimensions: [number of data vectors]
    weighted_scores : array_like
        PC scores without bad data points
        Dimensions: [number of data vectors, number of eigen vectors]
    mean_dot_data : array_like
    Mean dotted by the data, excluding bad values.
        Dimensions: [number of data vectors]
    mean_scores : array_like
        Mean projected to eigen system (i.e., pc scores of the mean)
        Dimensions: [number of data vectors, number of eigen vectors]

    Returns
    -------
    new_weighted_scores : array_like
        weighted PC scores ammended by the normalisation
        Dimensions: [number of data vectors, number of eigen vectors]

    """
    new_weighted_scores = mean_dot_mean[:,None] * weighted_scores - mean_dot_data[:,None] * mean_scores

    return new_weighted_scores

def invert_matrix(matrix_array):
    """
    Wrapper to invert a matrix.

    Parameters
    ----------
    matrix_array : array_like
        A matrix to invert
        Dimensions: [number of data vectors, number of eigen vectors, number of eigen vectors]

    Returns
    -------
    matrix_array_inverted : array_like
        The interted matrix
        Dimensions: [number of data vectors, number of eigen vectors, number of eigen vectors]
    """
    matrix_array_inverted = np.linalg.inv(matrix_array)

    return matrix_array_inverted

def get_pc_scores(inverted_correlation_matrix, weighted_scores):
    """
    PC scores that minimise X_sq with a normalisation factor where:

        X_sq =  W(|x - N(m + Va)|^2)

    `W` is the window function.
    `a` are the pc scores (Dimensions [, number of eigen vectors])
    `V` are the eigen vectors (Dimensions [length of data vector, number of eigen vectors])
    `x` is the data with gaps (Dimensions [, length of data vector])
    `m` is the mean data vector (Dimensions [length of data vector])

    This eistimates the pc scores that account for gaps. The cost is
    that when the scores are used to fill data gaps, the data is no longer
    fully othoganal since we are using the eigen system within the data itself

    Parameters
    ----------
    inverted_correlation_matrix : array_like
        Matrix describing the correlation between the weighted eigen vectors
        Dimensions: [number of data vectors, number of eigen vectors, number of eigen vectors]
    weighted_scores : array_like
        weighted PC scores ammended by the normalisation
        Dimensions: [number of data vectors, number of eigen vectors]

    Returns
    -------
    principle_comp_scores : array_like
        The pc scores for weighted data and a weighted eigen system
        Dimensions : [number of data vectors, number of eigen vectors]
    """
    principle_comp_scores = np.matmul(inverted_correlation_matrix, weighted_scores[:,:,None])
    principle_comp_scores = principle_comp_scores.reshape(*inverted_correlation_matrix.shape[:2])

    return principle_comp_scores

def get_normalisation(mean_dot_data, mean_dot_mean, principle_comp_scores, mean_scores):
    """
    The normalisation factor that rescales the data that was not replaced to
    a good estimate for the correct scale.

    Parameters
    ----------
    mean_dot_data : array_like
    Mean dotted by the data, excluding bad values.
    mean_dot_mean : array_like
        Mean dotted by itself, excluding bad values.
        Dimensions: [number of data vectors]
    principle_comp_scores : array_like
        The pc scores for weighted data and a weighted eigen system
        Dimensions : [number of data vectors, number of eigen vectors]
    mean_scores : array_like
        Mean projected to eigen system (i.e., pc scores of the mean)
        Dimensions: [number of data vectors, number of eigen vectors]

    Returns
    -------
    normalisation : array_like
        The normalisation for the non-filled data to correct the scaling
        after adding data to the gapped data

    """
    normalisation = mean_dot_data[:,None] / (mean_dot_mean[:,None] + np.sum(principle_comp_scores * mean_scores, axis=1))

    return normalisation

def run_normgappy(error_array, data, mean_array, eigen_vectors):
    """
    Main wrapper function to calculate the pc scores and data normalisation
    that accounts for missing/bad data values.
    The code is vectorised so you can either run code on a single spectra
    or run on the entire data set.
    Inputted data and errors need to have two dimentions as a result
    of the vectorisation. For a single spectra, a dummy dimension can
    be added by inputting your 1d data as: data[None], error_array[None]

    Parameters
    ----------
    error_array : array_like
        Errors of the data where bad/gap values are equal to zero.
        Dimensions: [number of data vectors, length of data vector]
    data : array_like
        Data that contains gaps
        Dimensions: [number of data vectors, length of data vector]
    mean_array : array_like
        The mean vector of the data set
        Dimensions: [length of data vector]
    eigen_vectors : array_like
        Eigen vectors of the data
        Dimensions [length of data vector, number of eigen vectors]

    Returns
    -------
    principle_comp_scores : array_like
        The pc scores for weighted data and a weighted eigen system
        Dimensions : [number of data vectors, number of eigen vectors]
    normalisation : array_like
        The normalisation for the non-filled data to correct the scaling
        after adding data to the gapped data

    """
    #This removes the data vector (spectrum) if it only contains bad values
    good_data_vector = where_data_all_good(error_array)
    error_array = error_array[good_data_vector]
    data = data[good_data_vector]

    weights = get_weights(error_array)


    # Here I've used the symbol names for brevity
    Fpr, Mpr, E = get_dchi_sq_dnorm(weights, data, mean_array, eigen_vectors)

    F, M = get_dchi_sq_dscores(weights, data, eigen_vectors)

    F_new = get_ammended_weighted_scores(Mpr, F, Fpr, E)

    M_new = get_ammended_weigthed_correlation_matrix(Fpr, M, F, E)
    M_new_inverted = invert_matrix(M_new)



    principle_comp_scores = get_pc_scores(M_new_inverted, F_new)
    normalisation = get_normalisation(Fpr, Mpr, principle_comp_scores, E)

    return principle_comp_scores, normalisation

def main():
    pass

if __name__ == "__main__":
    main()

