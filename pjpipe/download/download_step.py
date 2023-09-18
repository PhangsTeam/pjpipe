import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
import warnings
from functools import partial

import astropy.units as u
import numpy as np
from astropy.table import unique, vstack
from astroquery.exceptions import NoResultsWarning
from astroquery.mast import Observations
from requests.exceptions import ConnectionError, ChunkedEncodingError
from stdatamodels.jwst import datamodels
from tqdm import tqdm

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def download(
    observations,
    products,
    max_retries=5,
):
    """Wrap around astroquery download with retry option

    Args:
        observations: astroquery observation instance
        products: List of products to download
        max_retries (int): Maximum number of retries before failing.
            Defaults to 5
    """

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
                raise Warning("Max retries exceeded!")

    return True


def parallel_verify_integrity(
    file,
):
    """Wrapper to parallelise checking file integrity

    Args:
        file: File to check
    """

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            im = datamodels.open(file)
        del im
        gc.collect()
        return True
    except:
        gc.collect()
        return file


warnings.simplefilter("error", NoResultsWarning)


class DownloadStep:
    def __init__(
        self,
        target=None,
        prop_id=None,
        download_dir=None,
        procs=1,
        radius=None,
        telescope="JWST",
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
        overwrite=False,
    ):
        """Query and download data from MAST

        Args:
            target: Target name. Defaults to None, which will use the proposal ID to figure out names
            prop_id: Proposal identifier. Defaults to None, but at least one of `target`, `prop_id` needs
                to be set
            download_dir: Where to download to. Defaults to current directory if not set
            procs: Number of processes to run in parallel. Defaults to 1
            radius: Radius to search from centre of target in arcmin. Defaults to None, which uses astroquery default
            telescope: Telescope to use. Defaults to 'JWST'
            instrument_name: Something like 'NIRCAM/IMAGE', to filter down what products you want. Defaults to None
                (no filtering)
            dataproduct_type: These are things like 'cube', 'image' etc. Defaults to None (no filtering)
            calib_level: Calibration levels to filter. Defaults to None (no filtering)
            extension: File extensions to filter by. Defaults to None (no filtering)
            do_filter: Whether to filter dataproduct_type/calib_level/extension. Defaults to True
            breakout_targets: If targets is None, this will either download the whole proposal to a single directory
                (False), or breakout each target name to its own directory (True). Defaults to False
            product_type: Product types to download. Defaults to None (no filtering)
            filter_gs: Whether to filter guide stars from the observations. Defaults to True
            login: For proprietary data, set to True. Defaults to False
            api_key: If login is True, supply API key here
            overwrite: If True, overwrites existing data. Defaults to False
        """

        if not target and not prop_id:
            raise Warning("Either one of target, prop_id should be specified!")

        if download_dir is None:
            download_dir = os.getcwd()

        if calib_level is None:
            calib_level = [
                1,
                2,
                3,
            ]

        if product_type is None:
            product_type = [
                "SCIENCE",
                "PREVIEW",
                "INFO",
                "AUXILIARY",
            ]

        # Make sure the radius is in arcmin
        if not isinstance(radius, u.Quantity) and radius is not None:
            radius = radius * u.arcmin

        if isinstance(telescope, str):
            telescope = [telescope]
        if isinstance(prop_id, str):
            prop_id = [prop_id]
        if isinstance(instrument_name, str):
            instrument_name = [instrument_name]
        if isinstance(dataproduct_type, str):
            dataproduct_type = [dataproduct_type]

        self.target = target
        self.download_dir = download_dir
        self.procs = procs
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
        self.login = login
        self.api_key = api_key
        self.product_type = product_type
        self.overwrite = overwrite

        self.obs_list = None

        self.observations = Observations()

    def do_step(self):
        """Run download step"""

        step_complete_file = os.path.join(
            self.download_dir, "download_step_complete.txt"
        )

        cwd = os.getcwd()

        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        os.chdir(self.download_dir)

        if self.overwrite:
            shutil.rmtree("mastDownload")

        if os.path.exists(step_complete_file):
            log.info("Step already run")
            os.chdir(cwd)
            return True

        if self.login:
            if self.api_key is None:
                self.api_key = input("Supply an API key: ")
            self.observations.login(token=self.api_key)

        self.run_archive_query()

        if self.obs_list is None:
            shutil.rmtree(self.download_dir)
            os.chdir(cwd)
            return False

        success = self.run_download()
        if not success:
            os.chdir(cwd)
            return False

        # Check that all the files work now, else stop to rerun
        success = self.verify_integrity()
        if not success:
            raise Warning(
                "Integrity verification failed. Files will have been deleted, so run again"
            )

        # Make a file to let us know we've already downloaded
        # these files
        with open(step_complete_file, "w+") as f:
            f.close()

        os.chdir(cwd)

        return True

    def run_archive_query(self):
        """Query archive, trimming down as requested on instrument etc."""

        log.info(f"Downloading to: {self.download_dir}")
        log.info(f"Beginning archive query:")
        log.info(f"-> Target: {self.target}")
        log.info(f"-> Telescope: {self.telescope}")
        log.info(f"-> Proposal ID: {self.prop_id}")
        log.info(f"-> Instrument name: {self.instrument_name}")
        log.info(f"-> Data product type: {self.dataproduct_type}")

        if self.target is None:
            # If we don't have a target, fall back to just querying the proposal ID
            self.obs_list = self.observations.query_criteria(proposal_id=self.prop_id)

        else:
            # Query by target
            if self.radius is None:
                self.obs_list = self.observations.query_object(self.target)
            else:
                self.obs_list = self.observations.query_object(
                    self.target, radius=self.radius
                )

        if np.all(self.obs_list["calib_level"] < 0):
            log.warning("No available data")
            self.obs_list = None
            return False

        self.obs_list = self.obs_list[self.obs_list["calib_level"] >= 0]

        # Filter down by various options
        if self.telescope is not None:
            idx = [obs["obs_collection"] in self.telescope for obs in self.obs_list]
            self.obs_list = self.obs_list[idx]

        if self.prop_id is not None:
            idx = [obs["proposal_id"] in self.prop_id for obs in self.obs_list]
            self.obs_list = self.obs_list[idx]

        if self.instrument_name is not None:
            idx = [
                obs["instrument_name"] in self.instrument_name for obs in self.obs_list
            ]
            self.obs_list = self.obs_list[idx]

        if self.dataproduct_type is not None:
            idx = [
                obs["dataproduct_type"] in self.dataproduct_type
                for obs in self.obs_list
            ]
            self.obs_list = self.obs_list[idx]

        if len(self.obs_list) == 0:
            log.warning("No available data")
            self.obs_list = None
            return False

        return True

    def run_download(
        self,
        max_retries=5,
    ):
        """Download a list of observations

        Args:
            max_retries (int): Maximum number of retries before failing.
                Defaults to 5
        """

        if self.target is None and self.breakout_targets:
            # Break out by targets
            targets = np.unique(self.obs_list["target_name"])

            for target in targets:
                obs_list = self.obs_list[self.obs_list["target_name"] == target]

                target = target.lower()
                target = target.replace(" ", "_")

                target_dir = os.path.join(self.download_dir, target)

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                os.chdir(target_dir)

                products = self.get_products(obs_list)

                success = False
                n_retries = 0

                downloaded_files = glob.glob(
                    os.path.join(
                        "*",
                        "*",
                        "*",
                        "*",
                    )
                )
                n_files = len(downloaded_files)

                log.info(f"Downloading {len(products)} files")

                if n_files == len(products):
                    log.info(f"All files already downloaded")
                    os.chdir(self.download_dir)
                    continue

                while not success and n_retries < max_retries:
                    download(self.observations, products)

                    n_retries += 1

                    downloaded_files = glob.glob(
                        os.path.join(
                            "*",
                            "*",
                            "*",
                            "*",
                        )
                    )
                    n_files = len(downloaded_files)

                    if n_files == len(products):
                        success = True
                        os.chdir(self.download_dir)

                    if n_retries == max_retries:
                        raise Warning("Max retries exceeded!")

        else:
            products = self.get_products(self.obs_list)

            n_retries = 0

            downloaded_files = glob.glob(
                os.path.join(
                    "*",
                    "*",
                    "*",
                    "*",
                )
            )
            n_files = len(downloaded_files)

            log.info(f"Downloading {len(products)} files")

            if n_files == len(products):
                log.info(f"All files already downloaded")
                return True

            while n_retries < max_retries:
                download(self.observations, products)

                n_retries += 1

                downloaded_files = glob.glob(
                    os.path.join(
                        "*",
                        "*",
                        "*",
                        "*",
                    )
                )
                n_files = len(downloaded_files)

                if n_files == len(products):
                    return True

            raise Warning("Max retries exceeded!")

        return True

    def get_products(
        self,
        obs_list,
    ):
        """Get products from an observations list

        Args:
            obs_list: List of observations to get products from
        """

        # Flatten down all the observations
        products = []

        log.info("Getting observations")

        for obs in tqdm(
            obs_list,
            ascii=True,
        ):
            try:
                product_list = self.observations.get_product_list(obs)
                products.append(product_list)
            except NoResultsWarning:
                log.warning(f"Data not available for {obs['obs_id']}")
                continue

        products = vstack(products)

        log.info(f"Found a total {len(products)} files. Performing cuts")

        # Filter out guide stars if requested
        if self.filter_gs:
            mask = np.char.find(products["dataURI"], "_gs-") == -1
            products = products[mask]

        # Perform filtering if requested
        if self.do_filter:
            products = self.observations.filter_products(
                products,
                calib_level=self.calib_level,
                productType=self.product_type,
                extension=self.extension,
            )

        # Finally, remove duplicates and sort
        if "dataURI" in products.colnames:
            products = unique(
                products,
                keys="dataURI",
            )
            products.sort("dataURI")

        return products

    def verify_integrity(self):
        """Check files have downloaded successfully"""

        log.info("Verifying file integrity")

        downloaded_files = glob.glob(
            os.path.join(self.download_dir, "*", "*", "*", "*.fits")
        )
        downloaded_files.sort()

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(downloaded_files)])

        bad_files = []

        with mp.get_context("fork").Pool(procs) as pool:
            for bad_file in tqdm(
                pool.imap_unordered(
                    partial(
                        parallel_verify_integrity,
                    ),
                    downloaded_files,
                ),
                ascii=True,
                desc="Verifying integrity",
                total=len(downloaded_files),
            ):
                bad_files.append(bad_file)

            pool.close()
            pool.join()
            gc.collect()

        # Filter out the good files (Trues)
        bad_files = [file for file in bad_files if file is not True]

        if len(bad_files) == 0:
            log.info("All files verified successfully")
            return True
        else:
            log.warning(f"Found {len(bad_files)} bad files, will remove:")
            for file in bad_files:
                log.warning(f"-> {file}")
                os.remove(file)
            return False
