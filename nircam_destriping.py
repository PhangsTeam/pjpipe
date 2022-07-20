import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
from scipy.ndimage import median_filter

DESTRIPING_METHODS = ['row_median', 'median_filter']


class NircamDestriper:

    def __init__(self,
                 hdu_name=None,
                 destriping_method='row_median'
                 ):
        """Row-by-row destriping of NIRCAM images"""

        if hdu_name is None:
            raise Warning('hdu_name should be defined!')

        if destriping_method not in DESTRIPING_METHODS:
            raise Warning('destriping_method should be one of %s' % DESTRIPING_METHODS)

        self.hdu_name = hdu_name
        self.hdu = fits.open(self.hdu_name)
        self.destriping_method = destriping_method

    def run_destriping(self):

        if self.destriping_method == 'row_median':
            self.run_row_median()
        elif self.destriping_method == 'median_filter':
            self.run_median_filter()
        else:
            raise NotImplementedError('Destriping method %s not yet implemented!' % self.destriping_method)

    def run_row_median(self,
                       sigma=3,
                       npixels=5,
                       max_iters=20,
                       ):
        """Calculate sigma-clipped median for each row. From Tom Williams."""

        zero_idx = np.where(self.hdu['SCI'].data == 0)
        self.hdu['SCI'].data[zero_idx] = np.nan

        mask = make_source_mask(self.hdu['SCI'].data, nsigma=sigma, npixels=npixels)

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

        self.hdu.writeto(self.hdu_name.replace('.fits', '_destriped.fits'),
                         overwrite=True)

    def run_median_filter(self,
                          scales=None,
                          ):
        """Run a series of filters over the row medians. From Mederic Boquien."""

        if scales is None:
            scales = [3, 7, 15, 31, 63, 127]

        zero_idx = np.where(self.hdu['SCI'].data == 0)

        for scale in scales:
            med = np.median(self.hdu['SCI'].data, axis=1)
            noise = med - median_filter(med, scale)
            self.hdu['SCI'].data -= noise[:, np.newaxis]

        self.hdu['SCI'].data[zero_idx] = 0

        self.hdu.writeto(self.hdu_name.replace('.fits', '_destriped.fits'),
                         overwrite=True)
