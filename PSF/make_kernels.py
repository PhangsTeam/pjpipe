import copy
import warnings

import numpy as np
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.io import fits
from astropy.modeling import models, fitting
from photutils.centroids import centroid_com
from scipy.ndimage import rotate, zoom

PIXEL_SCALE_NAMES = ['XPIXSIZE', 'CDELT1', 'CD1_1', 'PIXELSCL']

def profile(psf,bins=None, pixscale=1):
    
    i_cen = (psf.shape[0] - 1) / 2
    j_cen = (psf.shape[1] - 1) / 2

    ji, ii = np.meshgrid((np.arange(psf.shape[1]) - j_cen),
                        (np.arange(psf.shape[0]) - i_cen))
    dis = (ji**2 + ii**2)**0.5*pixscale
    
    if bins is None:
        guess_sigma = np.sum(ii[:,int(i_cen)]**2*psf[:,int(i_cen)]/np.sum(psf[:,int(i_cen)]))**0.5*pixscale
        extent = np.min([psf.shape[0]/2*pixscale, guess_sigma*5])
        bins = np.linspace(0, int(extent), int(extent/pixscale/2))
            
 
    bin_means = (np.histogram(dis, bins, weights=psf)[0] /
                 np.histogram(dis, bins)[0])

    norm_bins = (bins[:-1]+np.diff(bins)/2)
    return norm_bins, bin_means


def get_fwhm(psf, pixscale=1):

    psf=psf/np.nanmax(psf)
    norm_bins, bin_means = profile(psf,  pixscale=pixscale)

    hwhm = np.interp(0.5, bin_means[::-1]/np.nanmax(bin_means), norm_bins[::-1])
    fwhm = 2*hwhm
    return fwhm

def get_pixscale(hdu):
    """Get pixel scale from header.

    Checks HDU header for pixel scale keywords, and returns a pixel scale in arcsec. If no suitable keyword is found,
    will throw up an error.

    Args:
        hdu (astropy.fits.PrimaryHDU): HDU to get pixel scale for.

    Returns:
        pix_scale (float): Pixel scale in arcsec.

    Raises:
        Warning: If no suitable pixel scale keyword is found in header.

    """
    if isinstance(hdu, (fits.hdu.image.PrimaryHDU, fits.hdu.image.ImageHDU) ):
        header =hdu.header
    elif isinstance(hdu, fits.header.Header ):
        header =hdu
    
    for pixel_keyword in PIXEL_SCALE_NAMES:
        try:
            try:
                pix_scale = np.abs(float(header[pixel_keyword]))
            except ValueError:
                continue
            if pixel_keyword in ['CDELT1', 'CD1_1']:
                pix_scale *= 3600
            return pix_scale
        except KeyError:
            pass

    raise Warning('No pixel scale found')


def fit_2d_gaussian(data, pixscale=None):
    """Fit 2D Gaussian to PSF.

    """

    # Normalise to peak of 1
    data /= np.nanmax(data)

    # Use centre of image for first guess centre, meshgrid up to feed into modelling
    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2

    ji, ii = np.meshgrid((np.arange(data.shape[1]) - j_cen),
                         (np.arange(data.shape[0]) - i_cen))
    if pixscale is not None:
        ji *= pixscale
        ii *= pixscale

    # Set up model
    model = models.Gaussian2D(amplitude=1,
                              x_mean=0, y_mean=0,
                              x_stddev=1, y_stddev=1,
                              theta=0
                              )
    fitter = fitting.LevMarLSQFitter()
    fit = fitter(model, ji, ii, data)

    # Take FWHM from mean of the stddevs
    fwhm = 2.355 * (fit.x_stddev.value + fit.y_stddev.value) / 2

    return fwhm


def interp_nans(data, x_stddev=2):
    """Interpolate over any NaNs present in an image.

    Uses astropy.convolution interpolate_replace_nans to smooth over any gaps left in an image.

    Args:
        data (numpy.ndarray): Input data to interpolate NaNs over.
        x_stddev (int, optional): Standard deviation of the Gaussian kernel.
            Defaults to 2 (pixels).

    Returns:
        numpy.ndarray: The data with NaNs interpolated over

    """

    kernel = Gaussian2DKernel(x_stddev=x_stddev)

    image_interp = interpolate_replace_nans(data, kernel)

    return image_interp


def centroid(data):
    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2

    j_centroid, i_centroid = centroid_com(data)

    # Shift the PSF to centre it

    i_shift = int(np.round(i_cen - i_centroid))
    j_shift = int(np.round(j_cen - j_centroid))

    data = np.roll(data, i_shift, axis=0)
    data = np.roll(data, j_shift, axis=1)

    return data


def resample(data, source_pixscale, target_pixscale, interp_order=3):
    """Resample data from one pixel scale to another

    """

    data_resample = zoom(data, source_pixscale / target_pixscale, order=interp_order)

    # force odd-sized array - the kernel needs to be odd

    if data_resample.shape[0] % 2 == 0:
        data_resample = data_resample[:data_resample.shape[0] - 1, :]
    if data_resample.shape[1] % 2 == 0:
        data_resample = data_resample[:, :data_resample.shape[1] - 1]

    return data_resample


def circularise(data, rotations=14):
    """Circularise a PSF.

    Rotate the PSF a number of times, taking an iterative average each time. This serves to make the PSF rotationally
    invariant.

    Args:
        data:
        rotations:

    Returns:

    """

    for n in range(rotations, 0, -1):
        data_rotate = rotate(data, 360 / (2 ** n), order=3, reshape=False)
        data = 0.5 * (data + data_rotate)

    # Set anything outside the maximum radius contained within the whole square to be 0

    radius = np.min(data.shape) / 2
    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2

    ji, ii = np.meshgrid((np.arange(data.shape[1]) - j_cen),
                         (np.arange(data.shape[0]) - i_cen))

    ri = ji ** 2 + ii ** 2
    data[ri > radius ** 2] = 0

    # Round anything within machine uncertainty to 0

    data[np.abs(data) < np.finfo(float).eps] = 0

    return data


def resize(data, pixscale, grid_size_arcsec=None):
    """Resize data to optimized grid size.

    Args:
        data:
        pixscale:
        grid_size_arcsec:

    Returns:

    TODO:
        * Will fail on non-square arrays

    """

    if grid_size_arcsec is None:
        grid_size_arcsec = np.array([729, 729])

    grid_size_pix = grid_size_arcsec / pixscale
    
    if grid_size_pix[0] % 2 == 0:
        grid_size_pix += 1
    
    if np.all(data.shape > grid_size_pix):
        data_resized = trim(data, grid_size_pix)
    elif np.all(data.shape < grid_size_pix):
        data_resized = zero_pad(data, grid_size_pix)
    else:
        raise Warning('Only square arrays implemented!')

    return data_resized


def trim(data, shape):
    """Trim data

    Args:
        data:
        shape:

    Returns:

    """

    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(data.shape, dtype=int)
    dshape = imshape - shape

    if np.alltrue(imshape == shape):
        return data

    if np.any(dshape % 2 != 0):
        raise ValueError('Source and target shapes have different parity.')

    idx, idy = np.indices(shape)
    offx, offy = dshape // 2

    return data[idx + offx, idy + offy]


def zero_pad(data, shape):
    """Zero pad data

    Args:
        data:
        shape:

    Returns:

    """

    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(data.shape, dtype=int)
    dshape = shape - imshape

    if np.alltrue(imshape == shape):
        return data

    data_pad = np.zeros(shape)
    idx, idy = np.indices(imshape)

    if np.any(dshape % 2 != 0):
        raise ValueError('Source and target shapes have different parity.')
    offx, offy = dshape // 2

    data_pad[idx + offx, idy + offy] = data

    return data_pad


def high_pass_filter(data, fwhm, pixscale=0.1):
    """Short desc

    Args:
        data (np.ndarray): Data to high-pass filter.
        fwhm (float): FWHM, in arcsec.
        pixscale (float, optional): Pixel scale, in arcsec. Defaults to 0.1.

    Returns:

    """

    k_b = 8 * np.pi / (fwhm * pixscale)
    k_a = 0.9 * k_b
    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2

    ji, ii = np.meshgrid((np.arange(data.shape[1]) - j_cen),
                         (np.arange(data.shape[0]) - i_cen))
    ri = np.sqrt(ji ** 2 + ii ** 2) * pixscale

    # Create filter
    data_filter = np.zeros(data.shape)
    data_filter[ri <= k_a] = 1

    idx = np.where((k_a < ri) & (ri <= k_b))
    data_filter[idx] = np.exp(-(1.8249 * (ri[idx] - k_a) / (k_b - k_a)) ** 4)

    # Apply filter
    data_filtered = data_filter * data

    return data_filtered


def low_pass_filter(data, pixscale=0.1):
    """Low-pass filter.

    """
    # Calculate k_h as the first value for which the radial value of the FFT drops below 0.5% of the max value

    i_range = data.shape[0]
    j_range = data.shape[1]

    data_slice = data[
                 int((i_range - 1) / 2):,
                 int((j_range - 1) / 2)
                 ]
    data_slice_max = np.nanmax(data_slice)

    try:
        k_h = np.where(data_slice < 0.005 * data_slice_max)[0][0] * pixscale
    except IndexError:
        print('k_h too large. Something is probably up with kernel generation...')
        k_h = len(data_slice) * pixscale
    k_l = 0.7 * k_h

    i_cen = (data.shape[0] - 1) / 2
    j_cen = (data.shape[1] - 1) / 2

    ji, ii = np.meshgrid((np.arange(data.shape[1]) - j_cen),
                         (np.arange(data.shape[0]) - i_cen))
    ri = np.sqrt(ji ** 2 + ii ** 2) * pixscale

    # Create the low-pass filter
    data_filter = np.zeros(data.shape)

    idx = np.where((k_l < ri) & (ri <= k_h))
    data_filter[idx] = 0.5 * (1 + np.cos(np.pi * (ri[idx] - k_l) / (k_h - k_l)))

    data_filter[ri <= k_l] = 1

    return data_filter


def trim_kernel_energy(kernel, energy_tol=0.999):
    """ Trim kernel based on enclosed energy to speed up later convolutions/space requirements

    Args:
        kernel:
        energy_tol:

    Returns:

    TODO:
        * This could fail if the kernel is too small/too big. Keep an eye out

    """
    kernel_radius = int((kernel.shape[0] - 1) / 2)

    i_cen = (kernel.shape[0] - 1) / 2
    j_cen = (kernel.shape[1] - 1) / 2
    ji, ii = np.meshgrid((np.arange(kernel.shape[1]) - j_cen),
                         (np.arange(kernel.shape[0]) - i_cen))

    ri = np.sqrt(ji ** 2 + ii ** 2)

    total_kernel_energy = np.nansum(np.abs(kernel[ri <= kernel_radius]))

    for radius in range(kernel_radius):
        idx = np.where(ri <= radius)
        enclosed_energy = np.nansum(np.abs(kernel[idx]))
        frac_kernel_energy = enclosed_energy / total_kernel_energy
        if frac_kernel_energy >= energy_tol:
            break
    trim_shape = (radius * 2 + 1, radius * 2 + 1)

    kernel_trimmed = trim(kernel, trim_shape)

    return kernel_trimmed


class MakeConvolutionKernel:
    """Class to generate kernels following the Aniano 2011 algorithm.

    Args:

        * arg: something

    Attributes:

        * attribute: something

    """

    def __init__(self,
                 source_psf=None,
                 source_fwhm=None,
                 source_name='source',
                 source_pixscale=1,
                 target_psf=None,
                 target_fwhm=None,
                 target_name='target',
                 target_pixscale=1,
                 common_pixscale=0.2,
                 grid_size_arcsec=None,
                 verbose=False,
                 ):
        """
        test
        """
        if source_psf is None:
            raise Warning('original_psf should be defined')
        if target_psf is None:
            raise Warning('target_psf should be defined')

        if isinstance(source_psf, (fits.hdu.image.PrimaryHDU, fits.hdu.image.ImageHDU) ):
            # get input PSF from file
            self.source_psf = copy.deepcopy(source_psf.data)
            self.source_pixscale = get_pixscale(source_psf)
        elif isinstance(source_psf, np.ndarray):
            self.source_psf = copy.deepcopy(source_psf)
            self.source_pixscale = source_pixscale
           # print('source PSF from array')
        else:
            raise Warning('source_psf is in an unknown format')
        
        if isinstance(target_psf, (fits.hdu.image.PrimaryHDU, fits.hdu.image.ImageHDU) ):
            # get input PSF from file
            self.target_psf = copy.deepcopy(target_psf.data)
            self.target_pixscale = get_pixscale(target_psf)
        elif isinstance(target_psf, np.ndarray):
            self.target_psf = copy.deepcopy(target_psf)
            self.target_pixscale = target_pixscale
            #print('target PSF from array')
        else:
            raise Warning('target_psf is in an unknown format')

        

        if not source_fwhm:
            #print('source_fwhm not supplied. Fitting using 2D Gaussian')
            source_fwhm = fit_2d_gaussian(data=self.source_psf, pixscale=self.source_pixscale)
        if not target_fwhm:
            # print('target_fwhm not supplied. Fitting using 2D Gaussian')
            target_fwhm = fit_2d_gaussian(data=self.target_psf, pixscale=self.target_pixscale)

        if source_fwhm >= target_fwhm:
            raise Warning('Cannot create kernel from lower to higher resolution data!')

        self.source_fwhm = copy.deepcopy(source_fwhm)
        self.target_fwhm = copy.deepcopy(target_fwhm)

        self.source_name = copy.deepcopy(source_name)
        self.target_name = copy.deepcopy(target_name)

        self.common_pixscale = copy.deepcopy(common_pixscale)

        self.source_fourier = None
        self.target_fourier = None

        self.kernel_fourier = None
        self.kernel = None

        self.grid_size_arcsec = grid_size_arcsec

        self.verbose = verbose

    def make_convolution_kernel(self):
        """Short desc

        Long desc

        Returns:
            * etc
        """

        if self.verbose:
            print('Interpolating')

        # Interpolate over any NaNs in the PSFs
        self.source_psf = interp_nans(self.source_psf)
        self.target_psf = interp_nans(self.target_psf)

        if self.verbose:
            print('Resampling')

        # Put onto common pixel grid
        self.source_psf = resample(self.source_psf, self.source_pixscale, self.common_pixscale)
        self.target_psf = resample(self.target_psf, self.target_pixscale, self.common_pixscale)

        # plt.figure()
        # plt.imshow(self.source_psf, vmin=vmin, vmax=vmax)

        if self.verbose:
            print('Centroiding')

        # Centroid
        self.source_psf = centroid(self.source_psf)
        self.target_psf = centroid(self.target_psf)

        if self.verbose:
            print('Circularising')

        # Circularise
        self.source_psf = circularise(self.source_psf)
        self.target_psf = circularise(self.target_psf)

        if self.verbose:
            print('Resizing')

        # Resize
        self.source_psf = resize(self.source_psf, self.common_pixscale, grid_size_arcsec=self.grid_size_arcsec)
        self.target_psf = resize(self.target_psf, self.common_pixscale, grid_size_arcsec=self.grid_size_arcsec)

        # Normalise
        self.source_psf /= np.nansum(self.source_psf)
        self.target_psf /= np.nansum(self.target_psf)

        if self.verbose:
            print('FFTing')

        # We now move onto the FFT part. Fourier transform the PSFs - only take the real part

        self.source_fourier = np.real(np.fft.fft2(np.fft.ifftshift(self.source_psf)))
        self.target_fourier = np.real(np.fft.fft2(np.fft.ifftshift(self.target_psf)))

        # Make sure centre of FFT is in middle

        self.source_fourier = np.fft.fftshift(self.source_fourier)
        self.target_fourier = np.fft.fftshift(self.target_fourier)

        if self.verbose:
            print('Circularising')

        # Circularise
        self.source_fourier = circularise(self.source_fourier)
        self.target_fourier = circularise(self.target_fourier)

        if self.verbose:
            print('High-pass filter')

        # High-pass filter
        source_fourier_high_pass = high_pass_filter(self.source_fourier, self.source_fwhm, self.common_pixscale / 2)
        target_fourier_high_pass = high_pass_filter(self.target_fourier, self.target_fwhm, self.common_pixscale / 2)

        if self.verbose:
            print('Inverting')

        # Invert the source fourier, any infs go to 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            source_fourier_invert = source_fourier_high_pass ** -1
            inf_idx = np.where(~np.isfinite(source_fourier_invert))
            source_fourier_invert[inf_idx] = 0

        if self.verbose:
            print('Low-pass filter')

        # Low-pass filter
        source_fourier_low_pass = low_pass_filter(self.source_fourier, self.common_pixscale / 2)

        if self.verbose:
            print('Creating kernel')

        # FFT of convolution kernel
        self.kernel_fourier = target_fourier_high_pass * (source_fourier_low_pass * source_fourier_invert)
        self.kernel_fourier = np.fft.ifftshift(self.kernel_fourier)

        if self.verbose:
            print('IFFT-ing')

        # IFFT to kernel and round out any tiny computational errors
        self.kernel = np.fft.fftshift(np.real(np.fft.ifft2(self.kernel_fourier)))
        self.kernel[np.abs(self.kernel) <= np.finfo(float).eps] = 0

        if self.verbose:
            print('Centroiding')

        # Centroid again, just in case
        self.kernel = centroid(self.kernel)

        if self.verbose:
            print('Trimming kernel')

        # Trim kernel based on enclosed energy
        self.kernel = trim_kernel_energy(self.kernel)

        if self.verbose:
            print('Last little bits')

        # Finally, circularise kernel and normalise to peak of 1
        self.kernel = circularise(self.kernel)
        self.kernel /= np.nanmax(self.kernel)

    def write_out_kernel(self, outdir=None):
        """

        Returns:

        """

        file_name = outdir+'%s_to_%s.fits' % (self.source_name, self.target_name)

        # Build the fits file. Use 32bit precision to cut down space

        hdu = fits.PrimaryHDU(data=np.array(self.kernel, dtype=np.float32))

        hdu.header['BITPIX'] = -32

        hdu.header['CRPIX1'] = (self.kernel.shape[1] + 1) / 2
        hdu.header['CRPIX2'] = (self.kernel.shape[0] + 1) / 2

        hdu.header['CRVAL1'] = 0.00
        hdu.header['CRVAL2'] = 0.00

        hdu.header['CDELT1'] = - self.common_pixscale / 3600
        hdu.header['CDELT2'] = self.common_pixscale / 3600

        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'

        hdu.writeto(file_name, overwrite=True)
