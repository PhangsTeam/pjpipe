import glob
import os
import warnings
from datetime import datetime

import numpy as np
from astropy.table import unique, vstack
from astroquery.exceptions import NoResultsWarning
from astroquery.mast import Observations
from requests.exceptions import ConnectionError, ChunkedEncodingError
from tqdm import tqdm


def get_time():
    """Get current time as a string"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time


def download(observations, products, max_retries=5):
    """Wrap around astroquery download with retry option"""

    for product in products:

        retry = 0
        success = False

        while retry < max_retries and not success:

            try:
                observations.download_products(product)
                success = True
            except (ConnectionResetError, ConnectionError, ChunkedEncodingError):
                retry += 1

            if retry == max_retries:
                raise Warning('Max retries exceeded!')

    return True


warnings.simplefilter('error', NoResultsWarning)


class ArchiveDownload:

    def __init__(self,
                 target=None,
                 radius=None,
                 telescope='JWST',
                 prop_id=None,
                 instrument_name=None,
                 dataproduct_type=None,
                 calib_level=None,
                 extension=None,
                 do_filter=True,
                 breakout_targets=False,
                 product_type=None,
                 filter_gs=True,
                 login=False,
                 api_key=None,
                 verbose=False,
                 overwrite=False,
                 ):
        """Query and download data from MAST"""

        if not target and not prop_id:
            raise Warning('Either one of target, prop_id should be specified!')

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
        self.dataproduct_type = dataproduct_type
        self.calib_level = calib_level
        self.extension = extension
        self.do_filter = do_filter
        self.breakout_targets = breakout_targets
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
            print('[%s] -> Data product type: %s' % (get_time(), self.dataproduct_type))

        if self.target is None:
            # If we don't have a target, fall back to just querying the proposal ID
            self.obs_list = self.observations.query_criteria(proposal_id=self.prop_id)

        else:
            # Query by target
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
        if self.dataproduct_type is not None:
            self.obs_list = self.obs_list[self.obs_list['dataproduct_type'] == self.dataproduct_type]

        if len(self.obs_list) == 0:
            print('[%s] No available data' % get_time())
            self.obs_list = None
            return False

        return True

    def run_download(self,
                     max_retries=5,
                     ):
        """Download a list of observations"""

        if self.target is None and self.breakout_targets:

            # Break out by targets
            targets = np.unique(self.obs_list['target_name'])

            base_dir = os.getcwd()

            for target in targets:

                obs_list = self.obs_list[self.obs_list['target_name'] == target]

                target = target.lower()
                target = target.replace(' ', '_')

                target_dir = os.path.join(base_dir, target)

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                os.chdir(target_dir)

                products = self.get_products(obs_list)

                success = False
                n_retries = 0

                downloaded_files = glob.glob(os.path.join('*',
                                                          '*',
                                                          '*',
                                                          '*')
                                             )
                n_files = len(downloaded_files)

                if n_files == len(products):
                    os.chdir(base_dir)
                    continue

                if self.verbose:
                    print('[%s] Downloading %d files' % (get_time(), len(products)))

                while not success and n_retries < max_retries:
                    download(self.observations, products)

                    n_retries += 1

                    downloaded_files = glob.glob(os.path.join('*',
                                                              '*',
                                                              '*',
                                                              '*')
                                                 )
                    n_files = len(downloaded_files)

                    if n_files == len(products):
                        success = True
                        os.chdir(base_dir)

                    if n_retries == max_retries:
                        raise Warning('Max retries exceeded!')

        else:

            products = self.get_products(self.obs_list)

            n_retries = 0

            downloaded_files = glob.glob(os.path.join('*',
                                                      '*',
                                                      '*',
                                                      '*')
                                         )
            n_files = len(downloaded_files)

            if n_files == len(products):
                return True

            if self.verbose:
                print('[%s] Downloading %d files' % (get_time(), len(products)))

            while n_retries < max_retries:
                download(self.observations, products)

                n_retries += 1

                downloaded_files = glob.glob(os.path.join('*',
                                                          '*',
                                                          '*',
                                                          '*')
                                             )
                n_files = len(downloaded_files)

                if n_files == len(products):
                    return True

            raise Warning('Max retries exceeded!')

        return True

    def get_products(self,
                     obs_list,
                     ):
        """Get products from an observations list"""

        # Flatten down all the observations
        products = []

        if self.verbose:
            print('[%s] Getting obs' % get_time())

        for obs in tqdm(obs_list, ascii=True):
            try:
                product_list = self.observations.get_product_list(obs)
                products.append(product_list)
            except NoResultsWarning:
                print('[%s] Data not available for %s' % (get_time(), obs['obs_id']))
                continue

        products = vstack(products)

        if self.verbose:
            print('[%s] Found a total %d files. Performing cuts...' % (get_time(), len(products)))

        # Filter out guide stars if requested
        if self.filter_gs:
            mask = np.char.find(products['dataURI'], '_gs-') == -1
            products = products[mask]

        # Perform filtering if requested
        if self.do_filter:
            products = self.observations.filter_products(products,
                                                         calib_level=self.calib_level,
                                                         productType=self.product_type,
                                                         extension=self.extension,
                                                         )

        # Finally, remove duplicates and sort
        if 'dataURI' in products.colnames:
            products = unique(products, keys='dataURI')
            products.sort('dataURI')

        return products
