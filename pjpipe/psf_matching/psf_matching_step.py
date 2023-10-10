import gc
import glob
import logging
import multiprocessing as mp
import os
import re
import shutil
import warnings
from functools import partial
from astropy.io import fits
from astropy.wcs import WCS


import numpy as np
from tqdm import tqdm
from astropy.convolution import convolve_fft
from scipy.interpolate import RegularGridInterpolator
from reproject import reproject_interp

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def get_pixscale(hdu):
    """ Get pixel scale from header. Checks HDU header
    for pixel scale keywords, and returns a pixel scale in arcsec. If
    no suitable keyword is found, will throw up an error.

    Args:

        hdu (astropy.fits.PrimaryHDU): HDU to get pixel scale for.

    Returns:

        pix_scale (float): Pixel scale in arcsec.

    Raises:

        Warning: If no suitable pixel scale keyword is found in header.

    Author:

    Tom Williams

    """

    PIXEL_SCALE_NAMES = ['XPIXSIZE', 'CDELT1', 'CD1_1', 'PIXELSCL']

    for pixel_keyword in PIXEL_SCALE_NAMES:
        try:
            try:
                pix_scale = np.abs(float(hdu.header[pixel_keyword]))
            except ValueError:
                continue
            if pixel_keyword in ['CDELT1', 'CD1_1']:
                pix_scale = WCS(hdu.header).proj_plane_pixel_scales()[0].value*3600
                # pix_scale *= 3600
            return pix_scale
        except KeyError:
            pass

    raise Warning('No pixel scale found')


def do_jwst_convolution(file_in, file_out, file_kernel,
                        blank_zeros=True,
                        output_grid=None,
                        ):
    """
    Convolves input image with an input kernel, and writes to disk.
    From Adam Leroy's script; adapted to process also errors and to do optional reprojection.
    :param file_in: path to image file
    :param file_out: path to output (convolved and optionally reprojected) file
    :param file_kernel: path to kernel for convolution
    :param blank_zeros: if True, than all zero values will be set to NaNs
    :param output_grid: None (no reprojection to be done) or tuple (wcs, shape) defining the grid for reprojection
    :return: None
    """
    with fits.open(file_kernel) as kernel_hdu:
        kernel_pix_scale = get_pixscale(kernel_hdu[0])
        # Note the shape and grid of the kernel as input
        kernel_data = kernel_hdu[0].data
        kernel_hdu_length = kernel_hdu[0].data.shape[0]
        original_central_pixel = (kernel_hdu_length - 1) / 2
        original_grid = (np.arange(kernel_hdu_length) - original_central_pixel) * kernel_pix_scale

    with fits.open(file_in) as image_hdu:
        if blank_zeros:
            # make sure that all zero values were set to NaNs, which
            # astropy convolution handles with interpolation
            image_hdu['ERR'].data[(image_hdu['SCI'].data == 0)] = np.nan
            image_hdu['SCI'].data[(image_hdu['SCI'].data == 0)] = np.nan

        image_pix_scale = get_pixscale(image_hdu['SCI'])

        # Calculate kernel size after interpolating to the image pixel
        # scale. Because sometimes there's a little pixel scale rounding
        # error, subtract a little bit off the optimum size (Tom
        # Williams).

        interpolate_kernel_size = np.floor(kernel_hdu_length * kernel_pix_scale / image_pix_scale) - 2

        # Ensure the kernel has a central pixel

        if interpolate_kernel_size % 2 == 0:
            interpolate_kernel_size -= 1

        # Define a new coordinate grid onto which to project the kernel
        # but using the pixel scale of the image

        new_central_pixel = (interpolate_kernel_size - 1) / 2
        new_grid = (np.arange(interpolate_kernel_size) - new_central_pixel) * image_pix_scale
        x_coords_new, y_coords_new = np.meshgrid(new_grid, new_grid)

        # Do the reprojection from the original kernel grid onto the new
        # grid with pixel scale matched to the image

        grid_interpolated = RegularGridInterpolator((original_grid, original_grid), kernel_data,
                                                    bounds_error=False, fill_value=0.)
        kernel_interp = grid_interpolated((x_coords_new.flatten(), y_coords_new.flatten()))
        kernel_interp = kernel_interp.reshape(x_coords_new.shape)

        # Ensure the interpolated kernel is normalized to 1
        kernel_interp = kernel_interp/np.nansum(kernel_interp)

        # Now with the kernel centered and matched in pixel scale to the
        # input image use the FFT convolution routine from astropy to
        # convolve.

        conv_im = convolve_fft(image_hdu['SCI'].data, kernel_interp,
                               allow_huge=True, preserve_nan=True,
                               fill_value=np.nan)

        # Convolve errors (with kernel**2, do not normalize it).
        # This, however, doesn't account for covariance between pixels
        conv_err = np.sqrt(convolve_fft(image_hdu['ERR'].data ** 2, kernel_interp ** 2,
                                        preserve_nan=True, allow_huge=True,
                                        normalize_kernel=False))

        image_hdu['SCI'].data = conv_im
        image_hdu['ERR'].data = conv_err

        if output_grid is None:
            image_hdu.writeto(file_out, overwrite=True)
        else:
            # Reprojection to target wcs grid define in output_grid
            target_wcs, target_shape = output_grid
            hdulist_out = fits.HDUList([fits.PrimaryHDU(header=image_hdu[0].header)])

            repr_data, fp = reproject_interp((conv_im, image_hdu['SCI'].header), output_projection=target_wcs,
                                             shape_out=target_shape)
            fp = fp.astype(bool)
            repr_data[~fp] = np.nan
            header = image_hdu['SCI'].header
            header.update(target_wcs.to_header())
            hdulist_out.append(fits.ImageHDU(data=repr_data, header=header, name='SCI'))

            # Note - this ignores the errors of interpolation and thus the resulting errors might be underestimated
            repr_err = reproject_interp((conv_err, image_hdu['SCI'].header), output_projection=target_wcs,
                                        shape_out=target_shape, return_footprint=False)
            repr_err[~fp] = np.nan
            header = image_hdu['ERR'].header
            hdulist_out.append(fits.ImageHDU(data=repr_err, header=header, name='ERR'))

            hdulist_out.writeto(file_out, overwrite=True)


class PSFMatchingStep:
    def __init__(
        self,
        target,
        in_dir,
        out_dir,
        procs,
        step_ext_in,
        kernels_dir,
        band=None,
        overwrite=False,
        target_bands=None,

    ):
        """Match PSF for all images"""

        self.target = target
        self.band = band
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.procs = procs
        self.step_ext_in = step_ext_in
        self.overwrite = overwrite
        self.kernels_dir = kernels_dir
        self.target_bands = target_bands

    def do_step(self):
        """Run psf_matching step"""

        if self.kernels_dir is None or not os.path.exists(self.kernels_dir):
            log.error("Kernels should be provided for psf-matching step. "
                      "Skip this step")
            return False

        step_complete_file = os.path.join(
            self.out_dir,
            "psf_matching_step_complete.txt",
        )

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        # Check if we've already run the step
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        files = glob.glob(
            os.path.join(
                self.in_dir,
                f"*_{self.step_ext_in}.fits",
            )
        )
        files.sort()

        procs = np.nanmin([self.procs, len(files)*len(self.target_bands)])

        successes = self.run_step(files, procs=procs)

        if not np.all(successes):
            log.warning("Failures detected during psf matching")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_step(self, files, procs=1):
        """Wrap paralellism around applying psf matching

        Args:
            files: List of files to process
            procs: Number of parallel processes to run.
                Defaults to 1
        """

        log.info("Running psf matching")

        files_process = []
        target_band_process = []
        for f in files:
            files_process.extend([f]*len(self.target_bands))
            target_band_process.extend(self.target_bands)

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in pool.imap_unordered(
                partial(
                    self.parallel_psf_match,
                    current_band=self.band,
                ),
                zip(files_process, target_band_process)
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_psf_match(
        self,
        current_task,
        current_band=None
    ):
        """Parallelize psf matching to target resolution

        Args:
            current_task: tuple (file, target_band),
                where file is the File to apply psf matching,
                and target_band is the band of target resolution
            current_band: band of the current image

        Returns:
            True or False
        """
        file, target_band = current_task
        target_band = target_band.upper()
        if target_band.startswith('GAUSS'):
            target_band = target_band.lower()
            target_band = target_band.replace("gauss", "Gauss")
        file_short = os.path.split(file)[-1]
        file_short = file_short.replace(self.step_ext_in, f"{self.step_ext_in}_at{target_band}")
        output_file = os.path.join(self.out_dir, file_short)
        kernel_file = os.path.join(self.kernels_dir, f"{current_band.upper()}_to_{target_band}.fits")
        if not os.path.exists(kernel_file):
            if target_band.startswith('Gauss') and 'P' in target_band:
                kernel_file = os.path.join(self.kernels_dir,
                                           f"{current_band.upper()}_to_{target_band.replace('p', '.')}.fits")
                if not os.path.exists(kernel_file):
                    kernel_exists = False
                else:
                    kernel_exists = True
            else:
                kernel_exists = False
        else:
            kernel_exists = True

        if not kernel_exists:
            return False

        # Check if the image with target resolution exists in the current reduction.
        # If yes, then the convolved images will be reprojected to match the image at target band
        check_file = file.replace(current_band.upper(), target_band.upper())
        check_file = check_file.replace(current_band.lower(), target_band.lower())
        if os.path.exists(check_file):
            with fits.open(check_file) as hdu:
                output_grid = (WCS(hdu['SCI'].header), hdu['SCI'].data.shape)
        else:
            output_grid = None

        do_jwst_convolution(file, output_file, file_kernel=kernel_file, output_grid=output_grid)

        return True
