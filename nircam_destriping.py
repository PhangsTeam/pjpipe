import copy

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
from scipy.ndimage import median_filter

DESTRIPING_METHODS = ['row_median', 'median_filter']


class NircamDestriper:

    def __init__(self,
                 hdu_name=None,
                 hdu_out_name=None,
                 quadrants=True,
                 destriping_method='row_median',
                 median_filter_scales=None,
                 sigma=3,
                 npixels=3,
                 max_iters=20,
                 dilate_size=2,
                 just_sci_hdu=False,
                 ):
        """NIRCAM Destriping routines

        Contains a number of routines to destripe NIRCAM data. For now, it's unclear what the best way forward is (and
        indeed, it may be different for different datasets), so there are a number of routines

        Args:
            * hdu_name (str): Input name for fits HDU
            * hdu_out_name (str): Output name for HDU
            * quadrants (bool): Whether to split the chip into 512 pixel segments, and destripe each (mostly)
                separately. Defaults to True
            * destriping_method (str): Method to use for destriping. Allowed options are given by DESTRIPING_METHODS
            * median_filter_scales (list): Scales for median filtering
            * sigma (float): Sigma for sigma-clipping. Defaults to 3
            * npixels (int): Pixels to grow for masking. Defaults to 5
            * max_iters (int): Maximum sigma-clipping iterations. Defaults to 20
            * dilate_size (int): make_source_mask dilation size. Defaults to 2
            * just_sci_hdu (bool): Write full fits HDU, or just SCI? Useful for testing, defaults to False
        """

        if hdu_name is None:
            raise Warning('hdu_name should be defined!')

        if destriping_method not in DESTRIPING_METHODS:
            raise Warning('destriping_method should be one of %s' % DESTRIPING_METHODS)

        self.hdu_name = hdu_name
        self.hdu = fits.open(self.hdu_name)

        if hdu_out_name is None:
            hdu_out_name = self.hdu_name.replace('.fits', '_destriped.fits')
        self.hdu_out_name = hdu_out_name

        self.just_sci_hdu = just_sci_hdu

        self.destriping_method = destriping_method

        if median_filter_scales is None:
            median_filter_scales = [3, 7, 15, 31, 63, 127]
        self.median_filter_scales = median_filter_scales
        self.sigma = sigma
        self.npixels = npixels
        self.max_iters = max_iters
        self.dilate_size = dilate_size

        self.quadrants = quadrants

    def run_destriping(self):

        if self.destriping_method == 'row_median':
            self.run_row_median()
        elif self.destriping_method == 'median_filter':
            self.run_median_filter()
        else:
            raise NotImplementedError('Destriping method %s not yet implemented!' % self.destriping_method)

        if self.just_sci_hdu:
            self.hdu['SCI'].writeto(self.hdu_out_name,
                                    overwrite=True)
        else:
            self.hdu.writeto(self.hdu_out_name,
                             overwrite=True)

    def run_row_median(self,
                       ):
        """Calculate sigma-clipped median for each row. From Tom Williams."""

        zero_idx = np.where(self.hdu['SCI'].data == 0)
        self.hdu['SCI'].data[zero_idx] = np.nan

        mask = make_source_mask(self.hdu['SCI'].data,
                                nsigma=self.sigma,
                                npixels=self.npixels,
                                dilate_size=self.dilate_size,
                                )

        if self.quadrants:

            quadrant_size = int(self.hdu['SCI'].data.shape[1] / 4)

            quadrants = {}

            # Calculate medians and apply
            for i in range(4):
                quadrants[i] = {}

                quadrants[i]['data'] = self.hdu['SCI'].data[:, i * quadrant_size: (i + 1) * quadrant_size]
                quadrants[i]['mask'] = mask[:, i * quadrant_size: (i + 1) * quadrant_size]

                # Do a quick check, and throw up a warning if we've got more than 10% of rows highly masked
                row_percent_masked = np.sum(~quadrants[i]['mask'], axis=1) / quadrant_size
                quadrants[i]['highly_masked'] = np.where(row_percent_masked < 0.1)
                n_high_masked = len(quadrants[i]['highly_masked'][0])

                if n_high_masked > quadrant_size / 10:
                    print('%d rows highly masked. This may cause issues in the final destriped image' % n_high_masked)

                quadrants[i]['median'] = sigma_clipped_stats(quadrants[i]['data'],
                                                             mask=quadrants[i]['mask'],
                                                             sigma=self.sigma,
                                                             maxiters=self.max_iters,
                                                             axis=1,
                                                             )[1]

            # # For highly masked rows, take the average corrections before and after
            # for i in range(4):
            #     idx = quadrants[i]['highly_masked']
            #     if i == 0:
            #         quadrants[i]['median'][idx] = quadrants[i + 1]['median'][idx]
            #
            #     elif 0 < i < 3:
            #         quadrants[i]['median'][idx] = np.nanmean(np.vstack([quadrants[i - 1]['median'][idx],
            #                                                             quadrants[i + 1]['median'][idx]]),
            #                                                  axis=0)
            #
            #     else:
            #         quadrants[i]['median'][idx] = quadrants[i - 1]['median'][idx]

            # Apply this to the data
            for i in range(4):
                quadrants[i]['data_corr'] = \
                    (quadrants[i]['data'] - quadrants[i]['median'][:, np.newaxis]) + \
                    np.nanmedian(quadrants[i]['median'][:, np.newaxis])

            # Match quadrants in the overlaps
            for i in range(3):
                quadrants[i + 1]['data_corr'] += \
                    np.nanmedian(quadrants[i]['data_corr'][:, quadrant_size - 10:]) - \
                    np.nanmedian(quadrants[i + 1]['data_corr'][:, 0:10])

            # Reconstruct the data
            for i in range(4):
                self.hdu['SCI'].data[:, i * quadrant_size: (i + 1) * quadrant_size] = quadrants[i]['data_corr']

        else:

            median_arr = sigma_clipped_stats(self.hdu['SCI'].data,
                                             mask=mask,
                                             sigma=self.sigma,
                                             maxiters=self.max_iters,
                                             axis=1,
                                             )[1]

            self.hdu['SCI'].data -= median_arr[:, np.newaxis]

            # Bring everything back up to the median level
            self.hdu['SCI'].data += np.nanmedian(median_arr)

        self.hdu['SCI'].data[zero_idx] = 0

    def run_median_filter(self,
                          use_mask=True
                          ):
        """Run a series of filters over the row medians. From Mederic Boquien."""

        zero_idx = np.where(self.hdu['SCI'].data == 0)
        self.hdu['SCI'].data[zero_idx] = np.nan

        mask = None
        if use_mask:
            mask = make_source_mask(self.hdu['SCI'].data,
                                    nsigma=self.sigma,
                                    npixels=self.npixels,
                                    dilate_size=self.dilate_size,
                                    )

        if self.quadrants:

            quadrant_size = int(self.hdu['SCI'].data.shape[1] / 4)

            quadrants = {}

            # Calculate medians and apply
            for i in range(4):
                quadrants[i] = {}

                if use_mask:
                    quadrants[i]['data'] = np.ma.array(
                        self.hdu['SCI'].data[:, i * quadrant_size: (i + 1) * quadrant_size],
                        mask=mask[:, i * quadrant_size: (i + 1) * quadrant_size]
                    )
                else:
                    quadrants[i]['data'] = self.hdu['SCI'].data[:, i * quadrant_size: (i + 1) * quadrant_size]

                for scale in self.median_filter_scales:

                    med = np.nanmedian(quadrants[i]['data'], axis=1)
                    med[~np.isfinite(med)] = 0
                    noise = med - median_filter(med, scale)

                    if use_mask:
                        quadrants[i]['data'] = np.ma.array(quadrants[i]['data'].data - noise[:, np.newaxis],
                                                           mask=quadrants[i]['data'].mask)
                    else:
                        quadrants[i]['data'] -= noise[:, np.newaxis]

                # Remove the mask since we don't need it any more
                if use_mask:
                    quadrants[i]['data'] = quadrants[i]['data'].data

            # Match quadrants in the overlaps
            for i in range(3):
                quadrants[i + 1]['data'] += \
                    np.nanmedian(quadrants[i]['data'][:, quadrant_size - 10:]) - \
                    np.nanmedian(quadrants[i + 1]['data'][:, 0:10])

            # Reconstruct the data
            for i in range(4):
                self.hdu['SCI'].data[:, i * quadrant_size: (i + 1) * quadrant_size] = quadrants[i]['data']

        else:

            for scale in self.median_filter_scales:

                if use_mask:
                    data = np.ma.array(
                        self.hdu['SCI'].data,
                        mask=mask
                    )
                else:
                    data = copy.deepcopy(self.hdu['SCI'].data)

                med = np.nanmedian(data, axis=1)
                med[~np.isfinite(med)] = 0
                noise = med - median_filter(med, scale)
                self.hdu['SCI'].data -= noise[:, np.newaxis]

        self.hdu['SCI'].data[zero_idx] = 0
