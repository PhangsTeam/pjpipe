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
                 destriping_method='row_median',
                 quadrants=True
                 ):
        """NIRCAM Destriping routines

        Contains a number of routines to destripe NIRCAM data. For now, it's unclear what the best way forward is (and
        indeed, it may be different for different datasets), so there are a number of routines

        Args:
            * hdu_name (str): Input name for fits HDU
            * destriping_method (str): Method to use for destriping. Allowed options are given by DESTRIPING_METHODS
            * quadrants (bool): Whether to split the chip into 512 pixel segments, and destripe each (mostly)
                separately. Defaults to True.

        """

        if hdu_name is None:
            raise Warning('hdu_name should be defined!')

        if destriping_method not in DESTRIPING_METHODS:
            raise Warning('destriping_method should be one of %s' % DESTRIPING_METHODS)

        self.hdu_name = hdu_name
        self.hdu = fits.open(self.hdu_name)
        self.destriping_method = destriping_method
        self.quadrants = quadrants

    def run_destriping(self):

        if self.destriping_method == 'row_median':
            self.run_row_median()
        elif self.destriping_method == 'median_filter':
            self.run_median_filter()
        else:
            raise NotImplementedError('Destriping method %s not yet implemented!' % self.destriping_method)

        self.hdu.writeto(self.hdu_name.replace('.fits', '_destriped.fits'),
                         overwrite=True)

    def run_row_median(self,
                       sigma=3,
                       npixels=5,
                       max_iters=20,
                       ):
        """Calculate sigma-clipped median for each row. From Tom Williams."""

        zero_idx = np.where(self.hdu['SCI'].data == 0)
        self.hdu['SCI'].data[zero_idx] = np.nan

        mask = make_source_mask(self.hdu['SCI'].data, nsigma=sigma, npixels=npixels)

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
                                                             sigma=sigma,
                                                             maxiters=max_iters,
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
                    np.nanmedian(quadrants[i]['data_corr'][:, quadrant_size - 20:quadrant_size]) - \
                    np.nanmedian(quadrants[i + 1]['data_corr'][:, 0:20])

            # Reconstruct the data
            for i in range(4):
                self.hdu['SCI'].data[:, i * quadrant_size: (i + 1) * quadrant_size] = quadrants[i]['data_corr']

        else:

            median_arr = sigma_clipped_stats(self.hdu['SCI'].data,
                                             mask=mask,
                                             sigma=sigma,
                                             maxiters=max_iters,
                                             axis=1,
                                             )[1]

            self.hdu['SCI'].data -= median_arr[:, np.newaxis]

            # Bring everything back up to the median level
            self.hdu['SCI'].data += np.nanmedian(median_arr)

        self.hdu['SCI'].data[zero_idx] = 0

    def run_median_filter(self,
                          scales=None,
                          ):
        """Run a series of filters over the row medians. From Mederic Boquien."""

        if scales is None:
            scales = [3, 7, 15, 31, 63, 127]

        zero_idx = np.where(self.hdu['SCI'].data == 0)

        if self.quadrants:

            quadrant_size = int(self.hdu['SCI'].data.shape[1] / 4)

            quadrants = {}

            # Calculate medians and apply
            for i in range(4):
                quadrants[i] = {}

                quadrants[i]['data'] = self.hdu['SCI'].data[:, i * quadrant_size: (i + 1) * quadrant_size]

                for scale in scales:
                    med = np.median(quadrants[i]['data'], axis=1)
                    noise = med - median_filter(med, scale)
                    quadrants[i]['data'] -= noise[:, np.newaxis]

            # Match quadrants in the overlaps
            for i in range(3):
                quadrants[i + 1]['data'] += \
                    np.nanmedian(quadrants[i]['data'][:, quadrant_size - 20:quadrant_size]) - \
                    np.nanmedian(quadrants[i + 1]['data'][:, 0:20])

            # Reconstruct the data
            for i in range(4):
                self.hdu['SCI'].data[:, i * quadrant_size: (i + 1) * quadrant_size] = quadrants[i]['data']

        else:

            for scale in scales:
                med = np.median(self.hdu['SCI'].data, axis=1)
                noise = med - median_filter(med, scale)
                self.hdu['SCI'].data -= noise[:, np.newaxis]

        self.hdu['SCI'].data[zero_idx] = 0
