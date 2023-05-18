import copy
import logging
import os
import pickle
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import webbpsf
from astropy.nddata.bitmask import interpret_bit_flags, bitfield_to_boolean_mask
from astropy.stats import sigma_clipped_stats
from jwst import datamodels
from jwst.datamodels.dqflags import pixel
from lmfit import minimize, Parameters, fit_report
from scipy.ndimage import shift

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

ALLOWED_MODES = [
    'replace',
    'subtract',
]
ALLOWED_FIT_METHODS = [
    'radial',
    'image',
]
ALLOWED_INSTRUMENTS = [
    'nircam',
    'miri',
]


def radial_profile(data,
                   x_centre,
                   y_centre,
                   error=None,
                   binsize=0.1,
                   interp_nan=False,
                   ):
    """Calculate the azimuthal-average radial profile, and associated error"""

    # Calculate distance from (x_centre, y_centre)
    y, x = np.indices(data.shape)
    r = np.sqrt((x - x_centre) ** 2 + (y - y_centre) ** 2) / binsize
    r = r.astype(int)

    if error is None:
        error = np.ones_like(data)

    mask = np.logical_or(data == 0,
                         ~np.isfinite(data)
                         )

    # Flatten and remove any bad data
    r_flat = r[~mask]
    data_flat = data[~mask]
    err_flat = error[~mask]

    # Calculate the radial profile, ignoring the first bin which will always be 0
    bin_data = np.bincount(r_flat, weights=data_flat / err_flat ** 2)
    bin_err = np.bincount(r_flat)
    bin_r = np.bincount(r_flat, weights=err_flat ** -2)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r_prof = bin_data / bin_r
        err_prof = np.sqrt(bin_err / bin_r)

    # Interpolate any pesky NaNs
    if interp_nan:
        good_idx = np.isfinite(r_prof)
        r_arr = np.arange(r_flat.max() + 1)
        r_prof = np.interp(r_arr, r_arr[good_idx], r_prof[good_idx])
        err_prof = np.interp(r_arr, r_arr[good_idx], err_prof[good_idx])

    return r_prof, err_prof


def image_resid(theta,
                data=None,
                error=None,
                model=None,
                mask=None,
                x_cen=None,
                y_cen=None,
                ):
    """Scale and shift PSF, calculate residual.

    Optionally allow a mask to specify good data. The mask should be 0
    for good data

    """

    if data is None:
        raise TypeError('data should be defined!')
    if model is None:
        raise TypeError('model should be defined')

    amp = theta['amp']
    x_shift = theta['x_cen']
    y_shift = theta['y_cen']
    offset = theta['offset']

    if x_cen is not None:
        x_shift -= x_cen
    if y_cen is not None:
        y_shift -= y_cen

    # Shift, scale, and offset the model
    model_shift = amp * shift(model, [y_shift, x_shift]) + offset

    if mask is not None:
        data = data[mask == 0]
        model_shift = model_shift[mask == 0]

        if error is not None:
            error = error[mask == 0]

    resid = residual(data,
                     model_shift,
                     err=error,
                     )

    return resid


def rad_prof_resid(theta,
                   data=None,
                   error=None,
                   model=None,
                   binsize=0.1,
                   start_pix=10,
                   end_pix=None,
                   ):
    """Calculate a radial profile given parameters, scale PSF, and calculate residual"""

    if data is None:
        raise TypeError('data should be defined!')
    if model is None:
        raise TypeError('model should be defined')

    data_prof, err_prof = radial_profile(data,
                                         theta['x_cen'],
                                         theta['y_cen'],
                                         error=error,
                                         binsize=binsize,
                                         )
    model_scaled = model * theta['amp'] + theta['offset']

    # Focus on a cutout range, based on start_pix and end_pix
    if start_pix is None:
        start_idx = 0
    else:
        start_idx = int(start_pix // binsize)

    if end_pix is None:
        end_idx = None
    else:
        end_idx = int(end_pix // binsize)

    data_prof = data_prof[start_idx:end_idx]
    err_prof = err_prof[start_idx:end_idx]
    model_scaled = model_scaled[start_idx:end_idx]

    # Make sure arrays are the same length
    max_len = np.min([len(data_prof), len(model_scaled)])
    data_prof = data_prof[:max_len]
    model_scaled = model_scaled[:max_len]
    err_prof = err_prof[:max_len]

    resid = residual(data_prof,
                     model_scaled,
                     err=err_prof,
                     )

    return resid


def residual(data,
             model,
             err=None,
             ):
    """Calculate residual for data (and optional error)"""

    # Filter NaNs
    good_idx = np.where(np.isfinite(data) & np.isfinite(model))
    data = data[good_idx]
    model = model[good_idx]

    if err is not None:
        err = err[good_idx]

    if err is None:
        return data - model
    else:
        return (data - model) / err


class PSFSubtraction:

    def __init__(self,
                 hdu_in_name,
                 hdu_out_name,
                 sat_coords,
                 instrument,
                 band,
                 fit_method='radial',
                 mode='replace',
                 binsize=1,
                 fit_dir=None,
                 plot_dir=None,
                 overwrite=False,
                 ):
        """PSF fitting/subtraction routine for saturated JWST pixels.

        This operates on _cal.fits files, so just before lv3. Given a list of saturated coordinates,
        extracts radial profiles to best fit the PSF to that profile. It will then either replace
        saturated pixels or subtract off the fitted PSF.

        Args:
            hdu_in_name (str): Name for the input HDU
            hdu_out_name (str): Output filename
            sat_coords (list): Should be list of [ra, dec] values for the centres of saturated pixel groups
            instrument (str): Either 'miri' or 'nircam'
            band (str): MIRI/NIRCam band
            fit_method (str, optional): Either radial (will use radial profile matching) or image (will fit directly
                to image). Defaults to 'radial'
            mode (str, optional): Either replace (replaces saturated pixels) or subtract (will subtract off the PSF)
            binsize (float, optional): Binsize to calculate the radial profile in. Defaults to 1
            fit_dir (str, optional): Directory to save fit results to. Defaults to None, which will not save
                anything
            plot_dir (str, optional): Directory to save diagnostic plots to. Defaults to None, which will not
                save anything
            overwrite (bool, optional): Whether to overwrite fits and corrected data. Defaults to False
        """

        self.hdu_in_name = hdu_in_name
        self.hdu_out_name = hdu_out_name

        if os.path.exists(self.hdu_out_name) and not overwrite:
            raise Warning('Output filename already exists and overwrite is not True!')

        self.hdu = datamodels.open(self.hdu_in_name)

        self.sat_coords = sat_coords

        self.instrument = instrument
        self.band = band

        if fit_method not in ALLOWED_FIT_METHODS:
            raise ValueError('fit_method should be one of %s' % ALLOWED_FIT_METHODS)
        self.fit_method = fit_method

        if mode not in ALLOWED_MODES:
            raise ValueError('mode should be one of %s' % ALLOWED_MODES)
        self.mode = mode

        self.binsize = binsize

        self.fit_dir = fit_dir
        self.plot_dir = plot_dir

        self.overwrite = overwrite

        self.psf = None
        self.psf_x_cen = None
        self.psf_y_cen = None
        self.psf_prof = None
        self.psf_model = None
        self.sat_bit_mask = None
        self.dq_bit_mask = None
        self.data_masked = None
        self.error_masked = None
        self.offset = None
        self.data_sub = None

    def run_fitting(self):

        self.prepare_data()

        for i, sat_coord in enumerate(self.sat_coords):

            if self.fit_method == 'radial':
                result = self.fit_psf_radial(sat_coord,
                                             sat_number=i,
                                             )
            elif self.fit_method == 'image':
                result = self.fit_psf_image(sat_coord,
                                            sat_number=i,
                                            )
            else:
                raise ValueError('fit_method should be one of %s' % ALLOWED_FIT_METHODS)

            if self.plot_dir is not None:
                self.plot_radial_fit(result,
                                     sat_number=i,
                                     )

            self.psf_model += self.shift_fit_psf(result)

        # We now have everything modelled! Mask out the bad data (but not saturation
        self.psf_model[np.logical_and(self.dq_bit_mask == 1,
                                      self.sat_bit_mask == 0)] = 0

        # First, replace the saturated pixels in the data
        self.hdu.data[self.sat_bit_mask == 1] = self.psf_model[self.sat_bit_mask == 1]

        if self.mode == 'subtract':
            # Now subtract data, but only for non-saturated pixels
            self.data_sub = self.hdu.data - self.psf_model
            self.data_sub[self.sat_bit_mask == 1] = self.hdu.data[self.sat_bit_mask == 1]

            # Add back in the offset to keep the background the same as when we came in
            self.data_sub += self.offset

        if self.plot_dir is not None:
            self.plot_models()

        if self.mode == 'subtract':
            self.hdu.data = copy.deepcopy(self.data_sub)

        # Make sure we've unflagged the saturated pixels now
        self.hdu.dq[self.sat_bit_mask] = 0

        # Save the data model back to disc
        self.hdu.save(self.hdu_out_name)

    def prepare_data(self):
        """Do a bunch of data prep before fitting

        - Get PSF (and radial profile)
        - Create DQ and saturation bitmasks
        - Mask data and error
        - Set up PSF model (just a constant offset to start
        """

        # Get the PSF, we want this to be as accurate to the observation date as possible
        self.psf = self.get_psf()
        psf_shape = np.array(self.psf.shape)
        self.psf_y_cen, self.psf_x_cen = psf_shape / 2 - 0.5

        # Get the PSF radial profile
        self.psf_prof, _ = radial_profile(self.psf,
                                          self.psf_x_cen,
                                          self.psf_y_cen,
                                          binsize=self.binsize,
                                          )

        # Create bit masks for saturation and good data
        self.sat_bit_mask = self.get_bit_mask(key='~SATURATED')
        self.dq_bit_mask = self.get_bit_mask(key='~DO_NOT_USE+NON_SCIENCE')

        self.psf_model = np.zeros_like(self.hdu.data)

        self.data_masked = copy.deepcopy(self.hdu.data)
        self.data_masked[self.dq_bit_mask == 1] = np.nan

        self.error_masked = copy.deepcopy(self.hdu.err)
        self.error_masked[self.dq_bit_mask == 1] = np.nan

        # Calculate the offset
        self.offset = sigma_clipped_stats(self.hdu.data,
                                          mask=self.dq_bit_mask,
                                          maxiters=None,
                                          )[1]

        self.psf_model += self.offset

    def fit_psf_image(self,
                      sat_coords,
                      sat_number=0,
                      ):
        """Fit the PSF to the saturated coordinates in image plane

        """

        x_cen, y_cen = self.hdu.meta.wcs.invert(sat_coords[0], sat_coords[1])
        x_cen = int(x_cen)
        y_cen = int(y_cen)

        init_amp_guess = self.get_initial_amp_guess(x_cen,
                                                    y_cen,
                                                    )

        # Create a base image that we'll shuffle around

        psf_model = np.zeros_like(self.hdu.data)
        psf_model[:self.psf.shape[0], :self.psf.shape[1]] = copy.deepcopy(self.psf)
        psf_model = shift(psf_model, [y_cen - self.psf_y_cen, x_cen - self.psf_x_cen])

        pars = Parameters()
        pars.add('amp',
                 value=init_amp_guess,
                 min=0.5 * init_amp_guess,
                 max=10 * init_amp_guess,
                 )
        pars.add('offset',
                 value=self.offset,
                 vary=False,
                 )

        # Here, x_cen and y_cen are offsets around 0, since we've already shuffled
        pars.add('x_cen',
                 value=x_cen,
                 min=x_cen - 5,
                 max=x_cen + 5,
                 )
        pars.add('y_cen',
                 value=y_cen,
                 min=y_cen - 5,
                 max=y_cen + 5,
                 )

        result_filename = None
        if self.fit_dir is not None:
            result_name = os.path.split(self.hdu_in_name)[-1].replace('.fits', '_im_fit_%s.pkl' % sat_number)

            result_filename = os.path.join(self.fit_dir,
                                           result_name,
                                           )

        if not os.path.exists(result_filename) or self.overwrite:
            result = minimize(image_resid,
                              pars,
                              args=(self.data_masked,
                                    self.error_masked,
                                    psf_model,
                                    self.dq_bit_mask,
                                    x_cen,
                                    y_cen,
                                    )
                              )

            if self.fit_dir is not None:
                with open(result_filename, 'wb') as f:
                    pickle.dump(result, f)

        else:
            with open(result_filename, 'rb') as f:
                result = pickle.load(f)

        logging.info(fit_report(result))

        return result

    def fit_psf_radial(self,
                       sat_coords,
                       sat_number=0,
                       ):
        """Fit the PSF to the saturated coordinates in a radial sense

        Because of the quantization of the radial profile, we do this in two steps.
        First fit using brute grid optimization to get the x- and y-centres, then
        a more refined fit to get the amplitude of the PSF.
        """

        x_cen, y_cen = self.hdu.meta.wcs.invert(sat_coords[0], sat_coords[1])
        x_cen = int(x_cen)
        y_cen = int(y_cen)

        init_amp_guess = self.get_initial_amp_guess(x_cen,
                                                    y_cen,
                                                    )

        # Do the fitting. Do a first pass with brute forcing in, to get the x and y centres

        pars = Parameters()
        pars.add('amp',
                 value=init_amp_guess,
                 min=0.5 * init_amp_guess,
                 max=10 * init_amp_guess,
                 )
        pars.add('offset',
                 value=self.offset,
                 vary=False,
                 )

        # Use a brute step here of the binsize
        pars.add('x_cen',
                 value=x_cen,
                 min=x_cen - 5,
                 max=x_cen + 5,
                 brute_step=self.binsize,
                 )
        pars.add('y_cen',
                 value=y_cen,
                 min=y_cen - 5,
                 max=y_cen + 5,
                 brute_step=self.binsize,
                 )

        result_filename = None
        if self.fit_dir is not None:
            result_name = os.path.split(self.hdu_in_name)[-1].replace('.fits', '_rad_fit_init_%d.pkl' % sat_number)

            result_filename = os.path.join(self.fit_dir,
                                           result_name,
                                           )

        if not os.path.exists(result_filename) or self.overwrite:
            result = minimize(rad_prof_resid,
                              pars,
                              method='brute',
                              args=(self.data_masked,
                                    self.error_masked,
                                    self.psf_prof,
                                    self.binsize,
                                    )
                              )

            if self.fit_dir is not None:
                with open(result_filename, 'wb') as f:
                    pickle.dump(result, f)

        else:
            with open(result_filename, 'rb') as f:
                result = pickle.load(f)

        logging.info(fit_report(result))

        # Pull out the best fit centres, and then fit again

        x_fit = result.params['x_cen'].value
        y_fit = result.params['y_cen'].value
        amp_fit = result.params['amp'].value

        pars = Parameters()
        pars.add('amp',
                 value=amp_fit,
                 )
        pars.add('offset',
                 value=self.offset,
                 vary=False,
                 )

        pars.add('x_cen',
                 value=x_fit,
                 vary=False,
                 )
        pars.add('y_cen',
                 value=y_fit,
                 vary=False,
                 )

        result_filename = None
        if self.fit_dir is not None:
            result_name = os.path.split(self.hdu_in_name)[-1].replace('.fits', '_rad_fit_final_%d.pkl' % sat_number)

            result_filename = os.path.join(self.fit_dir,
                                           result_name,
                                           )

        if not os.path.exists(result_filename) or self.overwrite:
            result = minimize(rad_prof_resid,
                              pars,
                              args=(self.data_masked,
                                    self.error_masked,
                                    self.psf_prof,
                                    self.binsize,
                                    )
                              )

            if self.fit_dir is not None:
                with open(result_filename, 'wb') as f:
                    pickle.dump(result, f)

        else:
            with open(result_filename, 'rb') as f:
                result = pickle.load(f)

        logging.info(fit_report(result))

        return result

    def get_initial_amp_guess(self,
                              x_cen,
                              y_cen,
                              min_pix=10,
                              scale=0.75,
                              ):
        """Get initial amplitude guess for PSF based on radial profiles"""

        # Get initial guess for the amplitude. Take radial profile
        init_data_prof, _ = radial_profile(self.data_masked,
                                           x_cen,
                                           y_cen,
                                           error=self.error_masked,
                                           binsize=self.binsize,
                                           )

        # We want to match either at some minimum point a few pixels out, or the first non-NaN value,
        # whichever is larger. This is to avoid the peaks of the PSF being deep in non-science territory
        min_point = int(min_pix // self.binsize)
        non_nan_point = np.where(~np.isnan(init_data_prof))[0][0]
        norm_point = np.max([min_point, non_nan_point])

        init_amp_guess = init_data_prof[norm_point] / self.psf_prof[norm_point]
        init_amp_guess = scale * init_amp_guess

        return init_amp_guess

    def get_psf(self,
                fov_pixels=511,
                ):
        """Get an odd-shaped PSF, given the input HDU parameters"""

        if self.instrument == 'miri':
            instrument = webbpsf.MIRI()
        elif self.instrument == 'nircam':
            instrument = webbpsf.NIRCam()
            instrument.detector = self.hdu.meta.instrument.detector
        else:
            raise Warning('instrument should be one of %s!' % ALLOWED_INSTRUMENTS)
        instrument.filter = self.hdu.meta.instrument.filter
        instrument.options['output_mode'] = 'detector sampled'

        instrument.load_wss_opd_by_date(self.hdu.meta.observation.date_beg)
        psf = instrument.calc_psf(fov_pixels=fov_pixels)

        # Pull out the data we care about, and normalise to peak of 1
        psf_data = copy.deepcopy(psf['DET_DIST'].data)
        psf_data /= np.nanmax(psf_data)

        return psf_data

    def get_bit_mask(self,
                     key,
                     ):
        """Get bit mask given bit names"""

        bits = interpret_bit_flags(key, flag_name_map=pixel)
        bit_mask = bitfield_to_boolean_mask(
            self.hdu.dq.astype(np.uint8),
            bits,
            good_mask_value=0,
            dtype=np.uint8
        )

        return bit_mask

    def shift_fit_psf(self,
                      result,
                      ):
        """Shift the PSF to match up to where it's been fit"""

        # Pull out best fit parameters
        x_fit = result.params['x_cen'].value
        y_fit = result.params['y_cen'].value
        amp_fit = result.params['amp'].value

        psf_shift = np.zeros_like(self.hdu.data)
        psf_shift[:self.psf.shape[0], :self.psf.shape[1]] = amp_fit * self.psf
        psf_shift = shift(psf_shift, [y_fit - self.psf_y_cen, x_fit - self.psf_x_cen])

        return psf_shift

    def plot_radial_fit(self,
                        result,
                        sat_number=0,
                        figsize=(6, 4),
                        fig_exts=None,
                        ):
        """Take an lmfit result and plot the radial profile

        """

        if fig_exts is None:
            fig_exts = ['png', 'pdf']

        # Pull out best fit parameters
        x_fit = result.params['x_cen'].value
        y_fit = result.params['y_cen'].value
        amp_fit = result.params['amp'].value
        offset_fit = result.params['offset'].value

        # Generate best-fit radial profile
        fit_data_prof, _ = radial_profile(self.data_masked,
                                          x_fit,
                                          y_fit,
                                          error=self.error_masked,
                                          binsize=self.binsize,
                                          )

        # Generate best-fit PSF profile
        fit_psf_prof = self.psf_prof * amp_fit + offset_fit

        plot_name = os.path.join(self.plot_dir,
                                 os.path.split(self.hdu_in_name)[-1].replace('.fits', '_radial_fit_%d' % sat_number)
                                 )

        # Plot these all up
        plt.figure(figsize=figsize)

        plt.plot(np.arange(len(fit_data_prof)) / self.binsize, fit_data_prof,
                 c='k',
                 label='Data',
                 )
        plt.plot(np.arange(len(fit_psf_prof)) / self.binsize, fit_psf_prof,
                 c='r',
                 label='PSF fit',
                 alpha=0.5,
                 )

        plt.legend(loc='upper right',
                   fancybox=False,
                   framealpha=1,
                   edgecolor='k',
                   )

        plt.xlim([0, len(fit_psf_prof) / self.binsize])

        plt.xlabel('R (pix)')
        plt.ylabel('Flux')

        plt.grid()

        plt.tight_layout()

        for fig_ext in fig_exts:
            plt.savefig('%s.%s' % (plot_name, fig_ext),
                        bbox_inches='tight',
                        )

    def plot_models(self,
                    figsize=(9, 4),
                    pmin=1,
                    pmax=99,
                    fig_exts=None,
                    ):
        """Plot the data (saturation replaced), PSF model and optionally the subtracted"""

        if fig_exts is None:
            fig_exts = ['png', 'pdf']

        if self.mode == 'subtract':
            n_subplots = 3
        elif self.mode == 'replace':
            n_subplots = 2
        else:
            raise Warning('mode should be one of replace, subtract!')

        vmin, vmax = np.nanpercentile(self.hdu.data, [pmin, pmax])

        plot_name = os.path.join(self.plot_dir,
                                 os.path.split(self.hdu_in_name)[-1].replace('.fits', '_model')
                                 )

        plt.figure(figsize=figsize)

        plt.subplot(1, n_subplots, 1)

        plt.imshow(self.hdu.data,
                   vmin=vmin,
                   vmax=vmax,
                   origin='lower',
                   interpolation='none',
                   )
        plt.title('Data (sat. replaced)')

        plt.axis('off')

        plt.subplot(1, n_subplots, 2)

        plt.imshow(self.psf_model,
                   vmin=vmin,
                   vmax=vmax,
                   origin='lower',
                   interpolation='none',
                   )
        plt.title('Model')

        plt.axis('off')

        if self.mode == 'subtract':
            plt.subplot(1, n_subplots, 3)

            plt.imshow(self.data_sub,
                       vmin=vmin,
                       vmax=vmax,
                       origin='lower',
                       interpolation='none',
                       )

            plt.title('Subtracted')

            plt.axis('off')

        plt.subplots_adjust(hspace=0, wspace=0)

        for fig_ext in fig_exts:
            plt.savefig('%s.%s' % (plot_name, fig_ext),
                        bbox_inches='tight',
                        )
