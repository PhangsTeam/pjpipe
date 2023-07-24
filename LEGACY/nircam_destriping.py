import copy
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata.bitmask import interpret_bit_flags, bitfield_to_boolean_mask
from astropy.stats import sigma_clipped_stats, SigmaClip
from jwst.datamodels.dqflags import pixel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.segmentation import detect_threshold, detect_sources
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation
from skimage import filters

import LEGACY.pca.vwpca as vw
import LEGACY.pca.vwpca_normgappy as gappy

DESTRIPING_METHODS = [
    'row_median',
    'median_filter',
    'remstripe',
    'pca',
]

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


def make_source_mask(data,
                     mask=None,
                     background=None,
                     nsigma=3,
                     npixels=3,
                     dilate_size=11,
                     sigclip_iters=5,
                     ):
    """Make a source mask from segmentation image"""

    sc = SigmaClip(sigma=nsigma,
                   maxiters=sigclip_iters,
                   )
    threshold = detect_threshold(data,
                                 background=background,
                                 mask=mask,
                                 nsigma=nsigma,
                                 sigma_clip=sc,
                                 )

    segment_map = detect_sources(data,
                                 threshold,
                                 npixels=npixels,
                                 )

    # If sources are detected, we can make a segmentation mask, else fall back to 0 array
    try:
        mask = segment_map.make_source_mask(size=dilate_size)
    except AttributeError:
        mask = np.zeros(data.shape, dtype=bool)

    return mask


def butterworth_filter(data,
                       data_std=None,
                       dq_mask=None,
                       return_high_sn_mask=False,
                       ):
    """Butterworth filter data, accounting for bad data"""

    data = copy.deepcopy(data)

    if data_std is None:
        data_std = sigma_clipped_stats(data,
                                       mask=dq_mask,
                                       )[2]

    # Make a high signal-to-noise mask, because these guys will cause bad ol' negative bowls
    high_sn_mask = make_source_mask(data,
                                    # nsigma=10,
                                    nsigma=5,
                                    npixels=1,
                                    dilate_size=1,
                                    )

    # Replace bad data with random noise
    idx = np.where(np.isnan(data) | dq_mask | high_sn_mask)
    data[idx] = np.random.normal(loc=0, scale=data_std, size=len(idx[0]))

    # Pad out the data by reflection to avoid ringing at boundaries
    data_pad = np.zeros([data.shape[0] * 2, data.shape[1] * 2])
    data_pad[:data.shape[0], :data.shape[1]] = copy.deepcopy(data)
    data_pad[-data.shape[0]:, -data.shape[1]:] = copy.deepcopy(data[::-1, ::-1])
    data_pad[-data.shape[0]:, :data.shape[1]] = copy.deepcopy(data[::-1, :])
    data_pad[:data.shape[0], -data.shape[1]:] = copy.deepcopy(data[:, ::-1])
    data_pad = np.roll(data_pad, axis=[0, 1], shift=[data.shape[0] // 2, data.shape[1] // 2])
    data_pad = data_pad[data.shape[0] // 4:-data.shape[0] // 4, data.shape[1] // 4:-data.shape[1] // 4]

    # Filter the image to remove any large scale structure.
    data_filter = filters.butterworth(data_pad,
                                      high_pass=True,
                                      )
    data_filter = data_filter[data.shape[0] // 4:-data.shape[0] // 4, data.shape[1] // 4:-data.shape[1] // 4]
    # data_filter[idx] = np.random.normal(loc=0, scale=data_std, size=len(idx[0]))

    # Get rid of the high S/N stuff, replace with median
    data_filter[idx] = np.nan
    data_filter_med = np.nanmedian(data_filter, axis=1)
    for col in range(data_filter.shape[0]):
        col_idx = np.where(np.isnan(data_filter[col, :]))
        data_filter[col, col_idx[0]] = data_filter_med[col]

    if return_high_sn_mask:
        return data_filter, high_sn_mask
    else:
        return data_filter


class NircamDestriper:

    def __init__(self,
                 hdu_name=None,
                 hdu_out_name=None,
                 quadrants=True,
                 vertical_subtraction=True,
                 destriping_method='row_median',
                 filter_diffuse=False,
                 large_scale_subtraction=False,
                 sigma=3,
                 npixels=3,
                 dilate_size=11,
                 max_iters=20,
                 median_filter_scales='',
                 pca_components=50,
                 pca_reconstruct_components=10,
                 pca_file=None,
                 just_sci_hdu=False,
                 plot_dir=None,
                 ):
        """NIRCAM Destriping routines

        Contains a number of routines to destripe NIRCAM data. For now, it's unclear what the best way forward is (and
        indeed, it may be different for different datasets), so there are a number of routines

        Args:
            * hdu_name (str): Input name for fits HDU
            * hdu_out_name (str): Output name for HDU
            * quadrants (bool): Whether to split the chip into 512 pixel segments, and destripe each (mostly)
                separately. Defaults to True
            * vertical_subtraction (bool): Perform sigma-clipped median column subtraction? Defaults to True
            * destriping_method (str): Method to use for destriping. Allowed options are given by DESTRIPING_METHODS
            * filter_diffuse (bool or list): Whether to perform high-pass filter on data, to remove diffuse, extended
                emission. If a list, will match to hdu_name and will be True if matched. Defaults to False, but should
                be set True for observations where emission fills the FOV
            * large_scale_subtraction (bool). Whether to mitigate for large-scale stripes remaining after the diffuse
                filtering. Defaults to False
            * sigma (float): Sigma for sigma-clipping. Defaults to 3
            * npixels (int): Pixels to grow for masking. Defaults to 5
            * dilate_size (int): make_source_mask dilation size. Defaults to 11
            * max_iters (int): Maximum sigma-clipping iterations. Defaults to 20
            * median_filter_scales (list): Scales for median filtering
            * pca_components (int): Number of PCA components to model. Defaults to 50
            * pca_reconstruct_components (int): Number of PCA components to use in reconstruction. Defaults to 10
            * pca_file (str): Path to save PCA model to (should be a .pkl file). If using quadrants, it will append
                the quadrant number accordingly automatically. Defaults to None, which means will not save out files
            * just_sci_hdu (bool): Write full fits HDU, or just SCI? Useful for testing, defaults to False
            * plot_dir (str): Directory to save diagnostic plots to. Defaults to None, which will not save any plots
        """

        if hdu_name is None:
            raise Warning('hdu_name should be defined!')

        if destriping_method not in DESTRIPING_METHODS:
            raise Warning('destriping_method should be one of %s' % DESTRIPING_METHODS)

        self.hdu_name = hdu_name
        self.hdu = fits.open(self.hdu_name, memmap=False)

        # Check if this is a subarray
        self.is_subarray = 'sub' in self.hdu[0].header['SUBARRAY'].lower()

        if self.is_subarray:
            # Force off quadrants if we're in subarray mode
            quadrants = False

        self.full_noise_model = None

        if hdu_out_name is None:
            hdu_out_name = self.hdu_name.replace('.fits', '_destriped.fits')
        self.hdu_out_name = hdu_out_name

        self.just_sci_hdu = just_sci_hdu

        self.quadrants = quadrants
        self.vertical_subtraction = vertical_subtraction

        self.destriping_method = destriping_method

        if median_filter_scales == '':
            median_filter_scales = [3, 7, 15, 31, 63, 127]
        self.median_filter_scales = median_filter_scales

        self.filter_diffuse = filter_diffuse
        self.large_scale_subtraction = large_scale_subtraction
        self.sigma = sigma
        self.npixels = npixels
        if max_iters == '':
            max_iters = None
        self.max_iters = max_iters
        self.dilate_size = dilate_size

        self.pca_components = pca_components
        self.pca_reconstruct_components = pca_reconstruct_components
        self.pca_file = pca_file

        self.plot_dir = plot_dir

    def run_destriping(self):

        if self.quadrants:
            self.hdu['SCI'].data = self.level_data(self.hdu['SCI'].data,
                                                   )

        self.full_noise_model = np.zeros_like(self.hdu['SCI'].data)

        # Do vertical subtraction, if requested
        if self.vertical_subtraction:
            self.full_noise_model += self.run_vertical_subtraction()

        if self.destriping_method == 'row_median':
            self.full_noise_model += self.run_row_median()
        elif self.destriping_method == 'median_filter':
            self.full_noise_model += self.run_median_filter()
        elif self.destriping_method == 'remstripe':
            self.full_noise_model += self.run_remstriping()
        elif self.destriping_method == 'pca':
            self.full_noise_model += self.run_pca_denoise()
        else:
            raise NotImplementedError('Destriping method %s not yet implemented!' % self.destriping_method)

        zero_idx = np.where(self.hdu['SCI'].data == 0)
        nan_idx = np.where(np.isnan(self.hdu['SCI'].data))

        self.hdu['SCI'].data -= self.full_noise_model

        self.hdu['SCI'].data[zero_idx] = 0
        self.hdu['SCI'].data[nan_idx] = np.nan

        if self.plot_dir is not None:
            self.make_destripe_plot()

        if self.just_sci_hdu:
            self.hdu['SCI'].writeto(self.hdu_out_name,
                                    overwrite=True)
        else:
            self.hdu.writeto(self.hdu_out_name,
                             overwrite=True)

        self.hdu.close()

        return self.full_noise_model

    def run_vertical_subtraction(self,
                                 ):
        """Median filter subtraction of columns (optional diffuse emission filtering)"""

        hdu_data = copy.deepcopy(self.hdu['SCI'].data)

        zero_idx = np.where(hdu_data == 0)

        hdu_data[zero_idx] = np.nan

        hdu_data -= self.full_noise_model

        mask = make_source_mask(hdu_data,
                                nsigma=self.sigma,
                                npixels=self.npixels,
                                dilate_size=self.dilate_size,
                                sigclip_iters=self.max_iters,
                                )

        dq_mask = self.get_dq_mask()

        mask = mask | dq_mask

        # Cut out the reference edge pixels if we're not in subarray mode
        if not self.is_subarray:
            data = copy.deepcopy(hdu_data[4:-4, 4:-4])
            mask = mask[4:-4, 4:-4]
            dq_mask = dq_mask[4:-4, 4:-4]
        else:
            data = copy.deepcopy(hdu_data)

        filter_diffuse = self.parse_parameter(self.filter_diffuse)
        if filter_diffuse:
            data, mask = self.get_filter_diffuse(mask=mask,
                                                 dq_mask=dq_mask,
                                                 data=data,
                                                 )

        full_noise_model = np.zeros_like(hdu_data)
        trimmed_noise_model = np.zeros_like(data)

        # Use median filtering to avoid noise and boundary issues
        data = np.ma.array(
            copy.deepcopy(data),
            mask=copy.deepcopy(mask)
        )

        for scale in self.median_filter_scales:

            med = np.ma.median(data, axis=0)
            mask_idx = np.where(med.mask)
            med = med.data
            med[mask_idx] = np.nan
            med[~np.isfinite(med)] = 0
            noise = med - median_filter(med, scale, mode='reflect')

            data -= noise[np.newaxis, :]

            trimmed_noise_model += noise[np.newaxis, :]

        # median_arr = sigma_clipped_stats(data,
        #                                  mask=mask,
        #                                  sigma=self.sigma,
        #                                  maxiters=self.max_iters,
        #                                  axis=0,
        #                                  )[1]
        #
        # trimmed_noise_model += median_arr[np.newaxis, :]
        #
        # Bring everything back up to the median level
        trimmed_noise_model -= np.nanmedian(trimmed_noise_model)

        if not self.is_subarray:
            full_noise_model[4:-4, 4:-4] = copy.deepcopy(trimmed_noise_model)
        else:
            full_noise_model = copy.deepcopy(trimmed_noise_model)

        return full_noise_model

    def run_remstriping(self,
                        ):
        """Destriping based on the CEERS remstripe routine

        Mask out sources, then collapse a median along x and y to remove stripes.

        """

        hdu_data = copy.deepcopy(self.hdu['SCI'].data)

        zero_idx = np.where(hdu_data == 0)

        hdu_data[zero_idx] = np.nan

        hdu_data -= self.full_noise_model

        quadrant_size = hdu_data.shape[1] // 4

        mask = make_source_mask(hdu_data,
                                nsigma=self.sigma,
                                npixels=self.npixels,
                                dilate_size=self.dilate_size,
                                sigclip_iters=self.max_iters,
                                )

        dq_mask = self.get_dq_mask()

        mask = mask | dq_mask

        # Cut out the reference edge pixels if we're not in subarray mode
        if not self.is_subarray:
            data = copy.deepcopy(hdu_data[4:-4, 4:-4])
            mask = mask[4:-4, 4:-4]
            dq_mask = dq_mask[4:-4, 4:-4]
        else:
            data = copy.deepcopy(hdu_data)

        filter_diffuse = self.parse_parameter(self.filter_diffuse)
        if filter_diffuse:
            data, mask = self.get_filter_diffuse(mask=mask,
                                                 dq_mask=dq_mask,
                                                 data=data,
                                                 )

        full_noise_model = np.zeros_like(hdu_data)
        trimmed_noise_model = np.zeros_like(data)

        if self.quadrants:

            # Calculate medians and apply
            for i in range(4):

                if not self.is_subarray:
                    if i == 0:
                        idx_slice = slice(0, quadrant_size - 4)
                    elif i == 3:
                        idx_slice = slice(1532, 2040)
                    else:
                        idx_slice = slice(i * quadrant_size - 4, (i + 1) * quadrant_size - 4)
                else:
                    idx_slice = slice(i * quadrant_size, (i + 1) * quadrant_size)

                data_quadrants = data[:, idx_slice]
                mask_quadrants = mask[:, idx_slice]

                # Collapse first along the y direction
                median_quadrants = sigma_clipped_stats(data_quadrants,
                                                       mask=mask_quadrants,
                                                       sigma=self.sigma,
                                                       maxiters=self.max_iters,
                                                       axis=1,
                                                       )[1]

                # Subtract this off the data, then collapse along x direction

                x_stripes = median_quadrants[:, np.newaxis] - np.nanmedian(median_quadrants)

                trimmed_noise_model[:, idx_slice] += x_stripes
                data_quadrants_1 = data_quadrants - x_stripes

                median_quadrants = sigma_clipped_stats(data_quadrants_1,
                                                       mask=mask_quadrants,
                                                       sigma=self.sigma,
                                                       maxiters=self.max_iters,
                                                       axis=0,
                                                       )[1]

                y_stripes = median_quadrants[np.newaxis, :] - np.nanmedian(median_quadrants)

                trimmed_noise_model[:, idx_slice] += y_stripes

        else:

            median_arr = sigma_clipped_stats(data,
                                             mask=mask,
                                             sigma=self.sigma,
                                             maxiters=self.max_iters,
                                             axis=1,
                                             )[1]

            trimmed_noise_model += median_arr[:, np.newaxis]

            # Bring everything back up to the median level
            trimmed_noise_model -= np.nanmedian(median_arr)

        if self.plot_dir is not None and mask is not None:
            self.make_mask_plot(data=data,
                                mask=mask,
                                filter_diffuse=filter_diffuse,
                                )

        if not self.is_subarray:
            full_noise_model[4:-4, 4:-4] = copy.deepcopy(trimmed_noise_model)
        else:
            full_noise_model = copy.deepcopy(trimmed_noise_model)

        return full_noise_model

    def run_pca_denoise(self,
                        ):
        """PCA-based de-noising

        Build a PCA model for the noise using the robust PCA implementation from Tamas Budavari and Vivienne Wild. We
        mask the data, optionally high-pass filter (Butterworth) to remove extended diffuse emission, and build the PCA
        model from there. pca_final_med_row_subtraction is on, it will do a final row-by-row median subtraction, to
        catch large-scale noise that might get filtered out.

        """

        hdu_data = copy.deepcopy(self.hdu['SCI'].data)

        zero_idx = np.where(hdu_data == 0)

        quadrant_size = hdu_data.shape[1] // 4

        hdu_data[zero_idx] = np.nan

        hdu_data -= self.full_noise_model

        mask = make_source_mask(hdu_data,
                                nsigma=self.sigma,
                                npixels=self.npixels,
                                dilate_size=self.dilate_size,
                                sigclip_iters=self.max_iters,
                                )

        dq_mask = self.get_dq_mask()

        mask = mask | dq_mask

        data = copy.deepcopy(hdu_data)
        err = copy.deepcopy(self.hdu['ERR'].data)
        original_mask = copy.deepcopy(mask)

        # Trim off the 0 rows/cols if we're using the full array

        if not self.is_subarray:
            data = data[4:-4, 4:-4]
            err = err[4:-4, 4:-4]
            dq_mask = dq_mask[4:-4, 4:-4]
            mask = mask[4:-4, 4:-4]

        data_mean, data_med, data_std = sigma_clipped_stats(data,
                                                            mask=mask,
                                                            sigma=self.sigma,
                                                            maxiters=self.max_iters,
                                                            )

        data -= data_med

        filter_diffuse = self.parse_parameter(self.filter_diffuse)
        if filter_diffuse:

            data_train, mask_train = self.get_filter_diffuse(mask=mask,
                                                             dq_mask=dq_mask,
                                                             data=data,
                                                             )

        else:

            data_train = copy.deepcopy(data)
            mask_train = copy.deepcopy(mask)

        if self.plot_dir is not None:
            self.make_mask_plot(data=data_train,
                                mask=mask_train,
                                filter_diffuse=filter_diffuse,
                                )

        if self.quadrants:

            noise_model_arr = np.zeros_like(data)

            # original_data = data_train[mask_train]
            data_train[mask_train] = np.nan

            data_med = np.nanmedian(data_train, axis=1)

            for i in range(4):

                if i == 0:
                    idx_slice = slice(0, quadrant_size - 4)
                elif i == 3:
                    idx_slice = slice(1532, 2040)
                else:
                    idx_slice = slice(i * quadrant_size - 4, (i + 1) * quadrant_size - 4)

                data_quadrant = copy.deepcopy(data_train[:, idx_slice])
                train_mask_quadrant = copy.deepcopy(mask_train[:, idx_slice])
                err_quadrant = copy.deepcopy(err[:, idx_slice])

                norm_factor = np.abs(np.diff(np.nanpercentile(data_quadrant, [16, 84]))[0])
                norm_median = np.nanmedian(data_quadrant)

                data_quadrant = (data_quadrant - norm_median) / norm_factor + 1
                err_quadrant /= norm_factor

                # Get NaNs out of the error map
                quadrant_nan_idx = np.where(np.isnan(err_quadrant))
                data_quadrant[quadrant_nan_idx] = np.nan
                err_quadrant[quadrant_nan_idx] = 0

                # Replace NaNd data with column median
                for col in range(data_quadrant.shape[0]):
                    idx = np.where(np.isnan(data_quadrant[col, :]))
                    data_quadrant[col, idx[0]] = (data_med[col] - norm_median) / norm_factor + 1

                # For places where this is all NaN, just 0 to avoid errors
                data_quadrant[np.isnan(data_quadrant)] = 0

                # data_quadrant[train_mask_quadrant] = 0
                err_quadrant[train_mask_quadrant] = 0

                if self.pca_file is not None:
                    pca_file = self.pca_file.replace('.pkl', '_amp_%d.pkl' % i)
                else:
                    pca_file = None

                if self.pca_file is not None and os.path.exists(pca_file):
                    with open(pca_file, 'rb') as f:
                        eigen_system_dict = pickle.load(f)
                else:
                    eigen_system_dict = self.fit_robust_pca(data_quadrant,
                                                            err_quadrant,
                                                            train_mask_quadrant,
                                                            )
                    if pca_file is not None:
                        with open(pca_file, 'wb') as f:
                            pickle.dump(eigen_system_dict, f)

                noise_model = self.reconstruct_pca(eigen_system_dict,
                                                   data_quadrant,
                                                   err_quadrant,
                                                   train_mask_quadrant)

                noise_model = (noise_model.T - 1) * norm_factor + norm_median
                noise_model_arr[:, idx_slice] = copy.deepcopy(noise_model)

            full_noise_model = np.full_like(hdu_data, np.nan)
            full_noise_model[4:-4, 4:-4] = copy.deepcopy(noise_model_arr)

        else:

            data_train[mask_train] = np.nan
            err_train = copy.deepcopy(err)

            # Remove NaNs
            train_nan_idx = np.where(np.isnan(err_train))
            data_train[train_nan_idx] = np.nan
            err_train[train_nan_idx] = 0

            data_med = np.nanmedian(data_train, axis=1)

            norm_median = np.nanmedian(data_train)
            norm_factor = median_abs_deviation(data_train, axis=None, nan_policy='omit')

            data_train = (data_train - norm_median) / norm_factor + 1
            err_train /= norm_factor

            # Replace NaNd data with column median
            for col in range(data_train.shape[0]):
                idx = np.where(np.isnan(data_train[col, :]))
                data_train[col, idx[0]] = (data_med[col] - norm_median) / norm_factor + 1

            # For places where this is all NaN, just 0 to avoid errors
            data_train[np.isnan(data_train)] = 0

            # data_train[mask_train] = 0
            err_train[mask_train] = 0

            if self.pca_file is not None and os.path.exists(self.pca_file):
                with open(self.pca_file, 'rb') as f:
                    eigen_system_dict = pickle.load(f)
            else:
                eigen_system_dict = self.fit_robust_pca(data_train,
                                                        err_train,
                                                        mask_train,
                                                        )
                if self.pca_file is not None:
                    with open(self.pca_file, 'wb') as f:
                        pickle.dump(eigen_system_dict, f)

            noise_model = self.reconstruct_pca(eigen_system_dict,
                                               data_train,
                                               err_train,
                                               mask_train)

            noise_model = (noise_model.T - 1) * norm_factor

            full_noise_model = np.full_like(hdu_data, np.nan)

            if self.is_subarray:
                full_noise_model = copy.deepcopy(noise_model)
            else:
                full_noise_model[4:-4, 4:-4] = copy.deepcopy(noise_model)

        # Centre the noise model around 0 to preserve flux
        noise_med = sigma_clipped_stats(full_noise_model,
                                        mask=original_mask,
                                        sigma=self.sigma,
                                        maxiters=self.max_iters,
                                        )[1]
        full_noise_model -= noise_med

        return full_noise_model

    def fit_robust_pca(self,
                       data,
                       err,
                       mask,
                       mask_column_frac=0.25,
                       min_column_frac=0.5,
                       ):

        # In low masked cases, take the data where less than mask_column_frac is masked. In highly masked cases, take
        # min_column_frac of data to ensure we have enough to fit.
        min_n_cols = int(data.shape[1] * min_column_frac)
        mask_sum = np.sum(mask, axis=0)
        low_masked_cols = len(np.where(mask_sum < mask_column_frac * data.shape[0])[0])
        n_cols = np.max([low_masked_cols, min_n_cols])

        mask_idx = np.argsort(mask_sum)
        data_low_emission = data[:, mask_idx[:n_cols]]
        err_low_emission = err[:, mask_idx[:n_cols]]

        # Roll around the axis to avoid learning where the mask is
        for i in range(1):
            roll_idx = np.random.randint(low=0, high=data.shape[0], size=data.shape[1])
            data_roll = np.roll(data_low_emission, shift=roll_idx, axis=0)
            err_roll = np.roll(err_low_emission, shift=roll_idx, axis=0)

            data_low_emission = np.hstack([data_low_emission, data_roll])
            err_low_emission = np.hstack([err_low_emission, err_roll])

        shuffle_idx = np.random.permutation(data_low_emission.shape[1])
        data_shuffle = copy.deepcopy(data_low_emission[:, shuffle_idx])
        err_shuffle = copy.deepcopy(err_low_emission[:, shuffle_idx])

        eigen_system_dict = vw.run_robust_pca(data_shuffle.T,
                                              errors=err_shuffle.T,
                                              amount_of_eigen=self.pca_components,
                                              save_extra_param=False,
                                              number_of_iterations=3,
                                              c_sq=0.787 ** 2,
                                              )

        return eigen_system_dict

    def reconstruct_pca(self,
                        eigen_system_dict,
                        data,
                        err,
                        mask
                        ):

        mean_array = eigen_system_dict['m']
        eigen_vectors = eigen_system_dict['U']

        eigen_reconstruct = eigen_vectors[:, :self.pca_reconstruct_components]

        data[mask] = 0
        err[mask] = 0

        scores, norm = gappy.run_normgappy(err.T,
                                           data.T,
                                           mean_array,
                                           eigen_reconstruct,
                                           )
        noise_model = (scores @ eigen_reconstruct.T) + mean_array

        return noise_model

    def run_row_median(self,
                       ):
        """Calculate sigma-clipped median for each row. From Tom Williams."""

        hdu_data = copy.deepcopy(self.hdu['SCI'].data)

        zero_idx = np.where(hdu_data == 0)
        hdu_data[zero_idx] = np.nan

        hdu_data -= self.full_noise_model

        mask = make_source_mask(hdu_data,
                                nsigma=self.sigma,
                                npixels=self.npixels,
                                dilate_size=self.dilate_size,
                                )

        dq_mask = self.get_dq_mask()

        mask = mask | dq_mask

        full_noise_model = np.zeros_like(hdu_data)

        filter_diffuse = self.parse_parameter(self.filter_diffuse)
        if filter_diffuse:
            data, mask = self.get_filter_diffuse(mask=mask,
                                                 dq_mask=dq_mask,
                                                 data=hdu_data,
                                                 )
        else:
            data = copy.deepcopy(hdu_data)

        if self.quadrants:

            quadrant_size = int(data.shape[1] / 4)

            # Calculate medians and apply
            for i in range(4):
                data_quadrants = data[:, i * quadrant_size: (i + 1) * quadrant_size]
                mask_quadrants = mask[:, i * quadrant_size: (i + 1) * quadrant_size]

                median_quadrants = sigma_clipped_stats(data_quadrants,
                                                       mask=mask_quadrants,
                                                       sigma=self.sigma,
                                                       maxiters=self.max_iters,
                                                       axis=1,
                                                       )[1]

                full_noise_model[:, i * quadrant_size: (i + 1) * quadrant_size] += median_quadrants[:, np.newaxis]
                full_noise_model[:, i * quadrant_size: (i + 1) * quadrant_size] -= np.nanmedian(median_quadrants)

        else:

            median_arr = sigma_clipped_stats(data,
                                             mask=mask,
                                             sigma=self.sigma,
                                             maxiters=self.max_iters,
                                             axis=1,
                                             )[1]

            full_noise_model += median_arr[:, np.newaxis]

            # Bring everything back up to the median level
            full_noise_model -= np.nanmedian(median_arr)

        if self.plot_dir is not None:
            self.make_mask_plot(data=data,
                                mask=mask,
                                filter_diffuse=filter_diffuse,
                                )

        return full_noise_model

    def run_median_filter(self,
                          use_mask=True
                          ):
        """Run a series of filters over the row medians. From Mederic Boquien."""

        hdu_data = copy.deepcopy(self.hdu['SCI'].data)

        zero_idx = np.where(hdu_data == 0)
        hdu_data[zero_idx] = np.nan

        hdu_data -= self.full_noise_model

        full_noise_model = np.zeros_like(hdu_data)

        mask = None
        dq_mask = None
        if use_mask:
            mask = make_source_mask(hdu_data,
                                    nsigma=self.sigma,
                                    npixels=self.npixels,
                                    dilate_size=self.dilate_size,
                                    )
            dq_mask = self.get_dq_mask()
            mask = mask | dq_mask

        # original_mask = copy.deepcopy(mask)

        filter_diffuse = self.parse_parameter(self.filter_diffuse)
        if filter_diffuse:
            data, mask = self.get_filter_diffuse(mask=mask,
                                                 dq_mask=dq_mask,
                                                 data=hdu_data,
                                                 )
        else:
            data = copy.deepcopy(hdu_data)

        if self.quadrants:

            quadrant_size = int(data.shape[1] / 4)

            # Calculate medians and apply
            for i in range(4):

                if use_mask:

                    data_quadrant = data[:, i * quadrant_size: (i + 1) * quadrant_size]
                    mask_quadrant = mask[:, i * quadrant_size: (i + 1) * quadrant_size]

                    data_quadrant = np.ma.array(data_quadrant, mask=mask_quadrant)
                else:
                    data_quadrant = data[:, i * quadrant_size: (i + 1) * quadrant_size]

                for scale in self.median_filter_scales:

                    if use_mask:
                        med = np.ma.median(data_quadrant, axis=1)
                        mask_idx = np.where(med.mask)
                        med = med.data
                        med[mask_idx] = np.nan
                    else:
                        med = np.nanmedian(data_quadrant, axis=1)

                    # Replace any remaining NaNs with the median
                    med[~np.isfinite(med)] = np.nanmedian(med)
                    noise = med - median_filter(med, scale, mode='reflect')

                    if use_mask:
                        data_quadrant = np.ma.array(data_quadrant.data - noise[:, np.newaxis],
                                                    mask=data_quadrant.mask)
                    else:
                        data_quadrant -= noise[:, np.newaxis]

                    full_noise_model[:, i * quadrant_size: (i + 1) * quadrant_size] += noise[:, np.newaxis]

        else:

            if use_mask:
                data = np.ma.array(
                    copy.deepcopy(data),
                    mask=copy.deepcopy(mask)
                )
            else:
                data = copy.deepcopy(data)

            for scale in self.median_filter_scales:

                if use_mask:
                    med = np.ma.median(data, axis=1)
                    mask_idx = np.where(med.mask)
                    med = med.data
                    med[mask_idx] = np.nan
                else:
                    med = np.nanmedian(data, axis=1)
                med[~np.isfinite(med)] = 0
                noise = med - median_filter(med, scale, mode='reflect')

                data -= noise[:, np.newaxis]

                full_noise_model += noise[:, np.newaxis]

        if self.plot_dir is not None and mask is not None:
            self.make_mask_plot(data=data,
                                mask=mask,
                                filter_diffuse=filter_diffuse,
                                )

        return full_noise_model

    def level_data(self,
                   data,
                   ):
        """Level overlaps in NIRCAM amplifiers"""

        quadrant_size = data.shape[1] // 4

        dq_mask = self.get_dq_mask()

        for i in range(3):
            quad_1 = data[:, i * quadrant_size: (i + 1) * quadrant_size][:, quadrant_size - 20:]
            dq_1 = dq_mask[:, i * quadrant_size: (i + 1) * quadrant_size][:, quadrant_size - 20:]
            quad_2 = data[:, (i + 1) * quadrant_size: (i + 2) * quadrant_size][:, :20]
            dq_2 = dq_mask[:, (i + 1) * quadrant_size: (i + 2) * quadrant_size][:, :20]

            quad_1[dq_1] = np.nan
            quad_2[dq_2] = np.nan

            med_1 = np.nanmedian(quad_1,
                                 axis=1,
                                 )
            med_2 = np.nanmedian(quad_2,
                                 axis=1,
                                 )
            diff = med_1 - med_2

            delta = sigma_clipped_stats(diff, maxiters=None)[1]
            data[:, (i + 1) * quadrant_size: (i + 2) * quadrant_size] += delta

        return data

    def get_filter_diffuse(self,
                           mask,
                           dq_mask,
                           data=None,
                           ):

        if data is None:
            data = copy.deepcopy(self.hdu['SCI'].data)

        data_mean, data_med, data_std = sigma_clipped_stats(data,
                                                            mask=mask,
                                                            sigma=self.sigma,
                                                            maxiters=self.max_iters,
                                                            )

        data_filter, high_sn_mask = butterworth_filter(data,
                                                       data_std=data_std,
                                                       dq_mask=dq_mask,
                                                       return_high_sn_mask=True)
        data = copy.deepcopy(data_filter)

        # Make a mask from this data. Create a positive mask
        # with a relatively strict size tolerance, to make sure
        # we don't just get random noise

        mask_pos = make_source_mask(
            data,
            mask=dq_mask,
            nsigma=self.sigma,
            npixels=10,
            dilate_size=self.dilate_size,
            sigclip_iters=self.max_iters,
        )

        # Create a negative mask with relatively strong
        # dilation, to make sure we catch all those negs
        mask_neg = make_source_mask(
            -data,
            mask=dq_mask | mask_pos,
            nsigma=self.sigma,
            npixels=self.npixels,
            sigclip_iters=self.max_iters,
        )

        mask = mask_pos | mask_neg | dq_mask

        return data, mask

    def get_dq_mask(self):

        dq_bits = interpret_bit_flags('~DO_NOT_USE+NON_SCIENCE', flag_name_map=pixel)

        dq_mask = bitfield_to_boolean_mask(
            self.hdu['DQ'].data.astype(np.uint8),
            dq_bits,
            good_mask_value=0,
        )

        dq_mask = dq_mask | \
                  ~np.isfinite(self.hdu['SCI'].data) | \
                  ~np.isfinite(self.hdu['ERR'].data) | \
                  (self.hdu['SCI'].data == 0)

        return dq_mask

    def parse_parameter(self,
                        param_to_parse,
                        ):

        if isinstance(param_to_parse, list):
            param = False
            for param_to_parse_val in param_to_parse:
                if param_to_parse_val in self.hdu_name:
                    param = True
        else:
            param = copy.deepcopy(param_to_parse)

        return param

    def make_mask_plot(self,
                       data,
                       mask,
                       filter_diffuse=False,
                       ):
        """Create mask diagnostic plot"""

        plot_name = os.path.join(self.plot_dir,
                                 self.hdu_out_name.split(os.path.sep)[-1].replace('.fits', '_filter+mask')
                                 )

        vmin, vmax = np.nanpercentile(data, [2, 98])
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(data, origin='lower', vmin=vmin, vmax=vmax, interpolation='none')

        plt.axis('off')

        if filter_diffuse:
            title = 'Filtered Data'
        else:
            title = 'Data'

        plt.title(title)

        plt.subplot(1, 2, 2)
        plt.imshow(mask, origin='lower', interpolation='none')

        plt.axis('off')

        plt.title('Mask')

        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')
        plt.close()

    def make_destripe_plot(self):
        """Create diagnostic plot for the destriping
        """

        nan_idx = np.where(np.isnan(self.hdu['SCI'].data))
        zero_idx = np.where(self.hdu['SCI'].data == 0)
        original_data = self.hdu['SCI'].data + self.full_noise_model
        original_data[zero_idx] = 0
        original_data[nan_idx] = np.nan

        plot_name = os.path.join(self.plot_dir,
                                 self.hdu_out_name.split(os.path.sep)[-1].replace('.fits', '_noise_model'),
                                 )

        vmin, vmax = np.nanpercentile(self.full_noise_model, [1, 99])
        vmin_data, vmax_data = np.nanpercentile(self.hdu['SCI'].data, [10, 90])

        plt.figure(figsize=(8, 4))

        ax = plt.subplot(1, 3, 1)
        im = plt.imshow(original_data,
                        origin='lower',
                        vmin=vmin_data, vmax=vmax_data,
                        interpolation='none',
                        )
        plt.axis('off')

        plt.title('Original Data')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0)

        plt.colorbar(im, cax=cax, label='MJy/sr', orientation='horizontal')

        ax = plt.subplot(1, 3, 2)
        im = plt.imshow(self.full_noise_model,
                        origin='lower',
                        vmin=vmin, vmax=vmax,
                        interpolation='none',
                        )
        plt.axis('off')

        plt.title('Noise Model')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0)

        plt.colorbar(im, cax=cax, label='MJy/sr', orientation='horizontal')

        ax = plt.subplot(1, 3, 3)
        im = plt.imshow(self.hdu['SCI'].data,
                        origin='lower',
                        vmin=vmin_data, vmax=vmax_data,
                        interpolation='none',
                        )
        plt.axis('off')

        plt.title('Destriped Data')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0)

        plt.colorbar(im, cax=cax, label='MJy/sr', orientation='horizontal')

        plt.subplots_adjust(hspace=0, wspace=0)

        # plt.show()

        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')
        plt.close()
