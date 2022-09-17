import os
import warnings
from datetime import datetime

import numpy as np
from astropy.table import unique
from astroquery.exceptions import NoResultsWarning
from astroquery.mast import Observations
from requests.exceptions import ConnectionError, ChunkedEncodingError


def get_time():
    """Get current time as a string"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time


def download(observations, product, filter_gs=True, max_retries=5):
    """Wrap around astroquery download with retry option"""
    retry = 0

    if filter_gs:
        mask = np.char.find(product['dataURI'], '_gs-') == -1
        product = product[mask]

    # Remove duplicate filenames, especially important for spectroscopic data
    if 'dataURI' in product.colnames:
        product = unique(product, keys='dataURI')

    while retry < max_retries:

        try:
            observations.download_products(product)
            return True
        except (ConnectionResetError, ConnectionError, ChunkedEncodingError):
            retry += 1

    raise Warning('Max retries exceeded!')


warnings.simplefilter('error', NoResultsWarning)


class ArchiveDownload:

    def __init__(self,
                 target=None,
                 radius=None,
                 telescope='JWST',
                 prop_id=None,
                 instrument_name=None,
                 calib_level=None,
                 extension=None,
                 do_filter=True,
                 product_type=None,
                 filter_gs=True,
                 login=False,
                 api_key=None,
                 verbose=False,
                 overwrite=False,
                 ):
        """Query and download data from MAST"""

        if not target:
            raise Warning('Target should be specified!')

        if calib_level is None:
            calib_level = [2, 3]

        if product_type is None:
            product_type = [
                'SCIENCE',
                'PREVIEW',
                'INFO',
                # 'AUXILIARY',
            ]

        self.target = target
        self.radius = radius
        self.telescope = telescope
        self.prop_id = prop_id
        self.instrument_name = instrument_name
        self.calib_level = calib_level
        self.extension = extension
        self.do_filter = do_filter
        self.filter_gs = filter_gs
        self.product_type = product_type
        self.verbose = verbose
        self.overwrite = overwrite

        self.obs_list = None

        self.observations = Observations()

        if login:
            if api_key is None:
                raise Warning('If logging in, supply an API key!')
            self.observations.login(token=api_key)

    def archive_download(self):
        """Run everything"""

        self.run_archive_query()

        if self.obs_list is None:
            return False

        if self.overwrite:
            os.system('rm -rf mastDownload')

        self.run_download()

    def run_archive_query(self):
        """Query archive, trimming down as requested on instrument etc."""

        if self.verbose:
            print('[%s] Beginning archive query:' % get_time())
            print('[%s] -> Target: %s' % (get_time(), self.target))
            print('[%s] -> Telescope: %s' % (get_time(), self.telescope))
            print('[%s] -> Proposal ID: %s' % (get_time(), self.prop_id))
            print('[%s] -> Instrument name: %s' % (get_time(), self.instrument_name))

        if self.radius is None:
            self.obs_list = self.observations.query_object(self.target)
        else:
            self.obs_list = self.observations.query_object(self.target, radius=self.radius)

        if np.all(self.obs_list['calib_level'] < 0):
            print('[%s] No available data' % get_time())
            self.obs_list = None
            return False

        self.obs_list = self.obs_list[self.obs_list['calib_level'] >= 0]

        if self.telescope is not None:
            self.obs_list = self.obs_list[self.obs_list['obs_collection'] == self.telescope]
        if self.prop_id is not None:
            self.obs_list = self.obs_list[self.obs_list['proposal_id'] == self.prop_id]
        if self.instrument_name is not None:
            self.obs_list = self.obs_list[self.obs_list['instrument_name'] == self.instrument_name]

        if len(self.obs_list) == 0:
            print('[%s] No available data' % get_time())
            self.obs_list = None
            return False

        return True

    def run_download(self):
        """Download a list of observations"""

        for obs in self.obs_list:
            try:
                product_list = self.observations.get_product_list(obs)
            except NoResultsWarning:
                print('[%s] Data not available for %s' % (get_time(), obs['obs_id']))
                continue

            if self.verbose:
                print('[%s] Downloading %s' % (get_time(), obs['obs_id']))

            if self.do_filter:
                products = self.observations.filter_products(product_list,
                                                             calib_level=self.calib_level,
                                                             productType=self.product_type,
                                                             extension=self.extension,
                                                             )
                if len(products) > 0:
                    download(self.observations, products, filter_gs=self.filter_gs)
                else:
                    print('[%s] Filtered data not available' % get_time())
                    continue
            else:
                download(self.observations, product_list, filter_gs=self.filter_gs)

        return True
