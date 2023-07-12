# JWST Scripts

# =======================================
# Using Input JWST PSFs
# Then using pypher to convert PSF to PSF
# From JWST to JWSt, or to Gaussian or to Moffat
# If you want to use it with JWST PSFs, you need first to
# get them (F200W.fits, F300M.fits, etc)

# v0.0.1 EE - Munich - 20/07/2022

# pypher
try:
    import pypher.pypher as ph
except ImportError:
    print("IMPORT WARNING: pypher is needed for cube_convolve")

# Astropy
from astropy.convolution import (Moffat2DKernel, Gaussian2DKernel, 
                                 convolve, convolve_fft)
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.io import fits as pyfits

# os
from os.path import join as joinpath

# List of JWST bands
jwst_psf = ["200W", "300M", "335M", "360M", "770W", "1000W", "1130W", "2100W"]
psf_prefix = "F"

# Constant for the size of psfs
ratfwhm = 50.
nstd = 20

def moffat_psf(fwhm, size, scale=None, n=1.0):
    """Moffat kernel. Returns a Moffat function array according to given
    input parameters. Using astropy Moffat2DKernel.

    Input
        fwhm (float): fwhm of the Moffat kernel, in arcsec.
        n (float): power index of the Moffat
        size (int numpy array): size of the requested kernel along each axis.
            If ``size'' is a scalar number the final kernel will be a square of
            side ``size''. If ``size'' has two element they must be in
            (y_size, x_size) order. In each case size must be an integer
            number of pixels.
        scale (float): pixel scale of the image [optional]

    Returns
        PSF with array and scale in arcsec
    """
    if scale is None:
        scale = fwhm / ratfwhm

    if n is None:
        print("ERROR[moffat_psf]: n cannot be None for Moffat")
        return None

    # Check the size nature (scalar and array or not)
    if np.isscalar(size):
        size = np.repeat(size, 2) + 1

    if len(size) > 2:
        print('ERROR[moffat_psf]: size must have at most two elements.')
        return None

    if size is None:
        # using a gaussian equivalent to derive the size
        size = int(fwhm * gaussian_fwhm_to_sigma / scale) * nstd

    if np.isscalar(size):
        size = np.repeat(size, 2)

    # Calculate the gamma power for the Moffat function
    gamma = fwhm / (2.0 * scale * np.sqrt(2.0**(1. / n) - 1.0))
    moffat_k = Moffat2DKernel(gamma, n, x_size=size[1], y_size=size[0])

    return PSF(moffat_k.array, scale, f"moff{fwhm:4.2f}")


def gaussian_psf(fwhm, size=None, scale=None, **kwargs):
    """Gaussian kernel.

    Input
        fwhm (float): fwhm of the Gaussian kernel, in arcsec.
        size (int, ndarray): size of the requested kernel along each axis.
            If ``size'' is a scalar number the final kernel will be a square of
            side ``size''. If ``size'' has two element they must be in
            (y_size, x_size) order. In each case size must be an integer number
            of pixels.
        scale (float): pixel scale of the image
        **kwargs: is there to absorb any additional parameter which could be
           provided (but won't be used).


    Returns
        PSF with array and scale in arcsec
    """
    if scale is None:
        scale = fwhm / ratfwhm

    # Compute sigma for the gaussian in pixels
    std = fwhm * gaussian_fwhm_to_sigma / scale

    if size is None:
        size = int(std) * nstd + 1

    if np.isscalar(size):
        size = np.repeat(size, 2)

    if len(size) > 2:
        print('ERROR[gaussian_psf]: size must have at most two elements.')

    gaussian_k = Gaussian2DKernel(std, x_size=size[1], y_size=size[0])

    return PSF(gaussian_k.array, scale, f"gaus{fwhm:4.2f}")


# List of kernel functions
dict_psf = {'gaussian': gaussian_psf, 'moffat': moffat_psf}

def psf2d(function='gaussian', size=None, fwhm=1., nmoffat=None, scale=None):
    """Create a model of the target PSF of the convolution. The target PSF does
    not vary as a function of wavelength, therefore the output is a 2D array.
    It uses either a gaussian or a moffat function.

    Parameters
        function: str, optional
            the function to model the target PSF. Only 'gaussian' or 'moffat'
            are accepted. Default: 'gaussian'
        fwhm: float
            the FWHM of the psf
        size: int, array-like
            the size of the final array. If ``size'' is a scalar number the
            psf will be a square of side ``size''. If ``size'' has two
            elements they must be in (y_size, x_size) order.
        nmoffat (float): Moffat power index. It must be defined if
        function = 'moffat'.
            Default: None
        scale: float, optional
            the spatial scale of the final psf

    Returns
        psf
    """
    if function in dict_psf:
        function_psf = dict_psf[function]
        psf = function_psf(fwhm, size, scale=scale, n=nmoffat)
        return psf
    else:
        print("ERROR[psf2d]: input function not part of the available ones"
              "({})".format(list(dict_psf.keys())))
        return None


def get_jwst_psf(band="200W", folder_jwst="", scale=None):
    """Get the PSF file in specified folder

    Input
        band (str): name of the band
        folder(str): name of the folder where the files are

    Returns
        fits file
    """
    name = f"{psf_prefix}{band}.fits"
    fullname = joinpath(folder_jwst, name)
    if not os.path.isfile(fullname):
        print(f"ERROR: no file {name} in folder {folder}")
        return None

    # Opening the fits file
    psf = pyfits.open(fullname)

    if scale is None:
        if "PIXELSCL" in psf[0].header:
            scale = psf[0].header['PIXELSCL']
        elif 'CD1_1' in psf[0].header:
            scale = psf[0].header['CD1_1'] * 3600.
        elif 'CDELT_1' in psf[0].header:
            scale = psf[0].header['CDELT1'] * 3600.
        else:
            print("ERROR: cannot find a scale in the fits file")
            return None

    return PSF(psf[0].data, scale, band)

class PSF(object):
    """Class to initialise a PSF
    """

    def __init__(self, psfarray, scale=1, name=""):
        """Initialise the class with an array (2d)
        and a scale (in arcsec). The name is something
        to add if you wish, and can be useful when
        converting psfs.
        """

        self.psf = psfarray
        self.scale = scale
        self.name = name

    def write(self, **kwargs):
        """A small wrapper around the writeto from 
        astropy fits.
        """

        h = pyfits.Header()
        h['PIXELSCL'] = self.scale
        hdu = pyfits.PrimaryHDU(self.psf, header=h)
        nameout = f"{self.name}.fits"
        print(f"Writing {nameout}")
        hdu.writeto(nameout, **kwargs)

def read_psf(psfname, folder_jwst="", **kwargs):
    """Read the psf, whether it's JWST or a function

    Input
        psfname: str
            Name of the PSF fits file
        folder_jwst: str
            Name of the folder where JWST PSFs (fits) are

    return
        psf, as a PSF class
    """

    folder_jwst = kwargs.pop("folder_jwst", "")
    if psfname in jwst_psf:
        return get_jwst_psf(psfname, folder_jwst=folder_jwst)
    else:
        if psfname in dict_psf:
            return psf2d(psfname, **kwargs)


def psf_to_psf(psf1, psf2, dict_psf1={}, dict_psf2={}, 
               writefits=True, overwrite=False, **kwargs):
    """Convert from a given psf to another
    Input argument can be a string as a name of a JWST band or 
    a function in ['gaussian', 'moffat']. In the latter case,
    input arguments should be given = fwhm (float, arcsec), 
    scale (float, arcsecond per pixel), and size (int, number of pixels).

    Input
        psf1: str
            Input PSF as either a JWST band as in the list provided in this file or
            "gaussian" or "moffat". If the latter 2, needs a dictionary with
            the input parameters (in dict_psf)
        psf2: str
            Same as psf1, but for the output psf.

        dict_psf1: dictionary
            Providing input parameters for the input psf.
        dict_psf2: dictionary
            See dict_psf2. Same for the output psf.

    Returns
        Transformation kernel as PSF


    """
    psf1 = read_psf(psf1, **dict_psf1)
    psf2 = read_psf(psf2, **dict_psf2)
    kernel = psfa_to_psfa(psf1.psf, psf2.psf, 
                          pixscale_source=psf1.scale, 
                          pixscale_target=psf2.scale, **kwargs)

    kernel.name = f"p{psf_prefix}{psf1.name}_to_{psf2.name}"
    if writefits:
        kernel.write(overwrite=overwrite)

    return kernel


def psfa_to_psfa(psf_source, psf_target, pixscale_source=0.2,
                  pixscale_target=0.2, angle_source=0., angle_target=0.,
                  reg_fact=1e-4, verbose=False, overwrite=False, **kwargs):
    """calculate the convolution kernel to move from one PSF to a target one.
    This is an adaptation of the main pypher script that it is meant to be used
    from the terminal.

    Input
        psf_source (ndarray): 2D PSF of the source image.
        psf_target (ndarray): target 2D PSF
        pixscale_source (float): pixel scale of the source PSF [0.2]
        pixscale_target (float): pixel scale of the target PSF [0.2]
        angle_source (float): position angle of the source PSF. [0]
        angle_target (float): position angle of the target PSF. [0]
        reg_fact (float): Regularisation parameter for the Wiener filter [1.e-4]
        verbose (bool): If True it prints more info on screen [False]

    Returns:
        kernel: a 2D kernel that convolved with the source PSF
            returns the target PSF

    """

    # Set NaNs to 0.0 for both input PSFs
    psf_source = np.nan_to_num(psf_source)
    psf_target = np.nan_to_num(psf_target)

    if verbose:
        print('Source PSF pixel scale: %.2f arcsec', pixscale_source)
        print('Target PSF pixel scale: %.2f arcsec', pixscale_target)

    # Rotate images (if necessary)
    if angle_source != 0.0:
        psf_source = ph.imrotate(psf_source, angle_source)
        if verbose:
            print('Source PSF rotated by %.2f degrees', angle_source)
    if angle_target != 0.0:
        psf_target = ph.imrotate(psf_target, angle_target)
        if verbose:
            print('Target PSF rotated by %.2f degrees', angle_target)

    # Normalize the PSFs if needed
    normalize_s = psf_source.sum()
    normalize_t = psf_target.sum()
    if np.allclose([normalize_s], [1.0], atol=1.e-3):
        psf_source /= normalize_s
        if verbose:
            print('Source PSF normalized with normalization '
                  'constant {:0.2f}'.format(normalize_s))
    if np.allclose([normalize_t], [1.0], atol=1.e-3):
        psf_target /= normalize_t
        if verbose:
            print('Target PSF normalized with normalization '
                  'constant {:0.2f}'.format(normalize_t))

    # Resample high resolution image to the low one
    if pixscale_source != pixscale_target:
        try:
            psf_source = ph.imresample(psf_source,
                                       pixscale_source,
                                       pixscale_target)
        except MemoryError:
            print('- COMPUTATION ABORTED -')
            print('The size of the resampled PSF would have '
                  'exceeded 10K x 10K')
            print('Please resize your image and try again')
            return None
        if verbose:
            print('Source PSF resampled to the target pixel scale')

    # check the new size of the source vs. the target
    if psf_source.shape > psf_target.shape:
        psf_source = ph.trim(psf_source, psf_target.shape)
    else:
        psf_source = ph.zero_pad(psf_source, psf_target.shape,
                                 position='center')

    kernel, _ = ph.homogenization_kernel(psf_target, psf_source,
                                         reg_fact=reg_fact)

    if verbose:
        print('Kernel computed using Wiener filtering and a regularisation '
              'parameter r = {:0.2e}'.format(reg_fact))

    return PSF(kernel, pixscale_target)


def find_scale(header):
    """A simple wrapper to find the scale of the image
    Not robust but useful
    """

    if 'CD1_1' in header:
        scale = header['CD1_1'] * 3600.
    elif 'PIXELSCL' in header:
        scale = header['PIXELSCL']
    else:
        scale = header['CDELT1']  * 3600.

    return scale
def compare_kernels(namepsf1, namepsf2):
    """ Comparing 2 kernel, given as their fits names
    Will show the two images with the same scale.

    Input
        namepsf1: str
            Name of the first psf
        namepsf2: str
            Name of the second psf
    """

    fig1 = figure(1, figsize=(12, 5))
    ax1 = plt.subplot(121)
    psf1 = pyfits.open(namepsf1)
    psf2 = pyfits.open(namepsf2)
    d1 = psf1[0].data
    d2 = psf2[0].data
    d1n = d1 / np.max(d1)
    d2n = d2 / np.max(d2)
    sc1_arc = find_scale(psf1[0].header)
    sc2_arc = find_scale(psf2[0].header)
    n1x, n1y = psf1[0].header['NAXIS1'], psf1[0].header['NAXIS2']
    n2x, n2y = psf2[0].header['NAXIS1'], psf2[0].header['NAXIS2']
    ext1 = np.array([-n1x, n1x, -n1y, n1y]) * sc1_arc / 2.
    ext2 = np.array([-n2x, n2x, -n2y, n2y]) * sc2_arc / 2.
    ax1.imshow(np.log10(d1n), extent=ext1, vmin=-2, vmax=0) 
    ax2 = plt.subplot(122, sharey=ax1)
    ax2.imshow(np.log10(d2n), extent=ext2, vmin=-2, vmax=0) 
    ax2.axis('equal')
    ax1.set_xlabel("arcsec", fontsize=16)
    ax1.set_ylabel("arcsec", fontsize=16)
    ax2.set_xlabel("arcsec", fontsize=16)
    ax1.set_title("Aniano", fontsize=24)
    ax2.set_title("Pypher",fontsize=24)

def plot_kernels(namepsf1, namepsf2):
    """ Plot the comparisong between two psf,
    given by their fits image names.
    """
    fig1 = figure(1, figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    psf1 = pyfits.open(namepsf1)
    psf2 = pyfits.open(namepsf2)
    d1 = psf1[0].data
    d2 = psf2[0].data
    d1n = d1 / np.max(d1)
    d2n = d2 / np.max(d2)
    sc1_arc = find_scale(psf1[0].header)
    sc2_arc = find_scale(psf2[0].header)
    n1x, n1y = psf1[0].header['NAXIS1'], psf1[0].header['NAXIS2']
    n2x, n2y = psf2[0].header['NAXIS1'], psf2[0].header['NAXIS2']
    ext1 = np.array([-n1x, n1x, -n1y, n1y]) * sc1_arc / 2.
    ext2 = np.array([-n2x, n2x, -n2y, n2y]) * sc2_arc / 2.
    x1 = np.linspace(ext1[0], ext1[1], n1x)
    x2 = np.linspace(ext2[0], ext2[1], n2x)
    clf()
    plot(x1, np.log10(d1n[n1x//2]), label=namepsf1)
    plot(x2, np.log10(d2n[n2x//2]), label=namepsf2)
    xlabel("arcsec", fontsize=16)
    legend()

def circularise(namepsf, ncirc=1000):
    """Brute force circularisation
    Input is the name of the fits images
    """
    from scipy.ndimage import rotate
    import copy
    psf = pyfits.open(namepsf)
    d = psf[0].data
    u = np.zeros_like(d)
    theta = np.linspace(0, 360., ncirc)
    for i in range(ncirc-1):
        print(i, end='\r')
        t = theta[i]
        rd = rotate(d, t, reshape=False)
        u += rd
    psf[0].data = u
    psf.writeto(f"c{namepsf}", overwrite=True)
    psf.close()


# Ignore the following function
# def go(cp=True, pypher=True):
#     if cp:
#         for psf1 in list_psf:
#             command = f"cp F{psf1}.fits pypher/F{psf1}p.fits"
#             print(command)
#             os.system(command)
#             f = pyfits.open(f"F{psf1}.fits")
#             command = f"addpixscl pypher/F{psf1}p.fits {f[0].header['PIXELSCL']}"
#             print(command)
#             os.system(command)
# 
#     if pypher:
#         for i in range(len(list_psf)-1):
#             psf1 = list_psf[i]
#             for k in range(i+1, len(list_psf)-1):
#                 psf2 = list_psf[k]
#                 command = f"pypher pypher/F{psf1}p.fits pypher/F{psf2}p.fits pypher/pF{psf1}_F{psf2}.fits"
#                 print(command)
#                 os.system(command)
