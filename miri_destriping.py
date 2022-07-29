import copy

import numpy as np
from astropy.io import fits
from photutils import make_source_mask
from scipy.ndimage import median_filter

DESTRIPING_METHODS = ['median_filter']


class MiriDestriper:

    def __init__(self,
                 hdu_name=None,
                 hdu_out_name=None,
                 destriping_method='row_median',
                 median_filter_scales=None,
                 sigma=3,
                 npixels=3,
                 max_iters=20,
                 dilate_size=11,
                 just_sci_hdu=False,
                 ):
        """MIRI Destriping routines

        Contains a number of routines to destripe MIRI data. For now, it's unclear what the best way forward is (and
        indeed, it may be different for different datasets), so there are a number of routines

        Args:
            * hdu_name (str): Input name for fits HDU
            * hdu_out_name (str): Output name for HDU
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

    def run_destriping(self):

        if self.destriping_method == 'median_filter':
            self.run_median_filter()
        else:
            raise NotImplementedError('Destriping method %s not yet implemented!' % self.destriping_method)

        if self.just_sci_hdu:
            self.hdu['SCI'].writeto(self.hdu_out_name,
                                    overwrite=True)
        else:
            self.hdu.writeto(self.hdu_out_name,
                             overwrite=True)

    def run_median_filter(self,
                          use_mask=True
                          ):
        """Run a series of filters over the row medians. From Mederic Boquien."""

        zero_idx = np.where(self.hdu['SCI'].data == 0)
        self.hdu['SCI'].data[zero_idx] = np.nan

        miri_corona_slice = slice(0, 300)
        miri_corona_vals = copy.deepcopy(self.hdu['SCI'].data[:, miri_corona_slice])

        self.hdu['SCI'].data[:, miri_corona_slice] = np.nan

        mask = None
        if use_mask:
            mask = make_source_mask(self.hdu['SCI'].data,
                                    nsigma=self.sigma,
                                    npixels=self.npixels,
                                    dilate_size=self.dilate_size,
                                    )
            mask = mask | ~np.isfinite(self.hdu['SCI'].data)
            mask = mask | self.hdu['DQ'].data != 0

        for scale in self.median_filter_scales:

            if use_mask:
                data = np.ma.array(
                    copy.deepcopy(self.hdu['SCI'].data),
                    mask=copy.deepcopy(mask)
                )
            else:
                data = copy.deepcopy(self.hdu['SCI'].data)

            # mask out the coronagraphs
            self.hdu['SCI'].data[:, miri_corona_slice] = np.nan

            if use_mask:
                med = np.ma.median(data, axis=1)
                mask_idx = np.where(med.mask)
                med = med.data
                med[mask_idx] = np.nan
            else:
                med = np.nanmedian(data, axis=1)
            med[~np.isfinite(med)] = 0

            noise = med - median_filter(med, scale)

            # Put the coronagraphs back in
            self.hdu['SCI'].data[:, miri_corona_slice] = miri_corona_vals

            self.hdu['SCI'].data -= noise[:, np.newaxis]

            # Update the coronagraph values for later
            miri_corona_vals = copy.deepcopy(self.hdu['SCI'].data[:, miri_corona_slice])

        self.hdu['SCI'].data[zero_idx] = 0
